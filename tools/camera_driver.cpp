// Read-only DirectShow capability/control inspection; no stream or settings changes.
#include <pybind11/pybind11.h>
#include <windows.h>
#include <dshow.h>
#include <dvdmedia.h>
#include <wrl/client.h>
#include <vector>
#include <stdexcept>
namespace py = pybind11;
using Microsoft::WRL::ComPtr;
struct ComSession {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    ComSession() { if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) throw std::runtime_error("COM initialization failed"); }
    ~ComSession() { if (SUCCEEDED(hr)) CoUninitialize(); }
};
std::string utf8(const wchar_t* s) {
    if (!s) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
    std::string r(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, s, -1, r.data(), n, nullptr, nullptr);
    if (n) r.pop_back();
    return r;
}
template<class T> py::dict control(T* api, long property) {
    py::dict d;
    long lo=0, hi=0, step=0, def=0, caps=0, value=0, flags=0;
    HRESULT r = api->GetRange(property, &lo, &hi, &step, &def, &caps);
    HRESULT g = api->Get(property, &value, &flags);
    d["range_hresult"] = (long)r; d["get_hresult"] = (long)g;
    if (SUCCEEDED(r)) {
        d["min"]=lo; d["max"]=hi; d["step"]=step; d["default"]=def; d["capability_flags"]=caps;
    }
    if (SUCCEEDED(g)) { d["value"]=value; d["flags"]=flags; d["auto_enabled"]=(flags & 1)!=0; }
    return d;
}
void free_media(AM_MEDIA_TYPE* mt) {
    if (!mt) return;
    CoTaskMemFree(mt->pbFormat);
    if (mt->pUnk) mt->pUnk->Release();
    CoTaskMemFree(mt);
}
py::list inspect_devices() {
    ComSession session;
    ComPtr<ICreateDevEnum> dev;
    if (FAILED(CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                               IID_PPV_ARGS(dev.GetAddressOf())))) throw std::runtime_error("Device enumerator unavailable");
    ComPtr<IEnumMoniker> devices;
    py::list output;
    if (dev->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, devices.GetAddressOf(), 0)!=S_OK) return output;
    ComPtr<IMoniker> moniker;
    int index=0;
    while (devices->Next(1, moniker.ReleaseAndGetAddressOf(), nullptr)==S_OK) {
        py::dict d; d["index"]=index++;
        ComPtr<IPropertyBag> bag;
        if (SUCCEEDED(moniker->BindToStorage(nullptr,nullptr,IID_PPV_ARGS(bag.GetAddressOf())))) {
            for (auto key : {L"FriendlyName", L"DevicePath"}) {
                VARIANT v; VariantInit(&v);
                if (SUCCEEDED(bag->Read(key,&v,nullptr)) && v.vt==VT_BSTR) d[py::str(utf8(key))]=utf8(v.bstrVal);
                VariantClear(&v);
            }
        }
        ComPtr<IBaseFilter> filter;
        HRESULT bind=moniker->BindToObject(nullptr,nullptr,IID_PPV_ARGS(filter.GetAddressOf()));
        d["bind_hresult"]=(long)bind;
        if (SUCCEEDED(bind)) {
            ComPtr<IAMCameraControl> camera;
            if (SUCCEEDED(filter.As(&camera))) d["exposure"]=control(camera.Get(),CameraControl_Exposure);
            ComPtr<IAMVideoProcAmp> amp;
            if (SUCCEEDED(filter.As(&amp))) {
                d["white_balance"]=control(amp.Get(),VideoProcAmp_WhiteBalance);
                d["gain"]=control(amp.Get(),VideoProcAmp_Gain);
            }
            py::list modes;
            ComPtr<IEnumPins> pins;
            if (SUCCEEDED(filter->EnumPins(pins.GetAddressOf()))) {
                ComPtr<IPin> pin;
                while(pins->Next(1,pin.ReleaseAndGetAddressOf(),nullptr)==S_OK) {
                    PIN_DIRECTION direction;
                    if (FAILED(pin->QueryDirection(&direction)) || direction!=PINDIR_OUTPUT) continue;
                    ComPtr<IAMStreamConfig> stream;
                    if (FAILED(pin.As(&stream))) continue;
                    int count=0, size=0;
                    if (FAILED(stream->GetNumberOfCapabilities(&count,&size)) || size<(int)sizeof(VIDEO_STREAM_CONFIG_CAPS)) continue;
                    std::vector<BYTE> buffer(size);
                    for (int i=0;i<count;i++) {
                        AM_MEDIA_TYPE* mt=nullptr;
                        HRESULT hr=stream->GetStreamCaps(i,&mt,buffer.data());
                        if (FAILED(hr) || !mt) { free_media(mt); continue; }
                        auto caps=reinterpret_cast<VIDEO_STREAM_CONFIG_CAPS*>(buffer.data());
                        BITMAPINFOHEADER* bmi=nullptr; REFERENCE_TIME interval=0;
                        if (mt->formattype==FORMAT_VideoInfo && mt->cbFormat>=sizeof(VIDEOINFOHEADER)) {
                            auto v=reinterpret_cast<VIDEOINFOHEADER*>(mt->pbFormat); bmi=&v->bmiHeader; interval=v->AvgTimePerFrame;
                        } else if (mt->formattype==FORMAT_VideoInfo2 && mt->cbFormat>=sizeof(VIDEOINFOHEADER2)) {
                            auto v=reinterpret_cast<VIDEOINFOHEADER2*>(mt->pbFormat); bmi=&v->bmiHeader; interval=v->AvgTimePerFrame;
                        }
                        if (bmi) {
                            py::dict m; m["width"]=bmi->biWidth; m["height"]=bmi->biHeight;
                            wchar_t guid[40]; StringFromGUID2(mt->subtype,guid,40); m["subtype"]=utf8(guid);
                            char code[5]={0}; for(int j=0;j<4;j++) code[j]=(char)((bmi->biCompression>>(8*j))&255);
                            m["fourcc"]=std::string(code); m["default_fps"]=interval>0 ? 1e7/interval : 0;
                            m["min_fps"]=caps->MaxFrameInterval>0 ? 1e7/caps->MaxFrameInterval : 0;
                            m["max_fps"]=caps->MinFrameInterval>0 ? 1e7/caps->MinFrameInterval : 0;
                            modes.append(m);
                        }
                        free_media(mt);
                    }
                }
            }
            d["modes"]=modes;
        }
        output.append(d);
    }
    return output;
}
PYBIND11_MODULE(_camera_driver,m) { m.def("inspect_devices", &inspect_devices); }
