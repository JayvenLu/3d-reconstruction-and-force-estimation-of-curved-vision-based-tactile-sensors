import cv2
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch
import numpy as np
import os
from collections import namedtuple

# # 定义输入结构
# InputSample = namedtuple('InputSample', ['rgb', 'bg', 'position'])

class Modelv2_ori(nn.Module):
    def __init__(self, use_pos=True, dropout=0.1, output_dims=None,nir_training=True):
        super(Modelv2_ori, self).__init__()
        if output_dims is None:
            # output_dims = [176, 187, 154] # 搜索得到
            output_dims = [256,128,64]
        self.use_pos = use_pos
        self.nir_training = nir_training
        in_channel = 3
        if self.nir_training:
            in_channel+=1
        if self.use_pos:
            # in_channel += 32
            in_channel += 2
        input_dim = in_channel
        layers = []
        for output_dim in output_dims:
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1))
            layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Conv2d(input_dim, 3, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, sample):
        if self.nir_training:
            if self.use_pos:
                # o0 = torch.cat((sample["rgb"] - sample["bg"], sample["pe"]), dim=1)
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir']- sample['bg_nir'],
                                sample["position"]), dim=1)
            else:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir'] - sample['bg_nir']), dim=1)
        else:
            if self.use_pos:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"], sample["position"]), dim=1)
            else:
                o0 = sample["rgb"] - sample["bg_rgb"]


        o = self.layers(o0)

        return o

class Model_quan_l4_32(nn.Module):
    def __init__(self, use_pos=True, dropout=0.4, output_dims=None, nir_training=True):
        super(Model_quan_l4_32, self).__init__()
        if output_dims is None:
            output_dims = [32, 32, 32]
        self.use_pos = use_pos
        self.nir_training = nir_training
        in_channel = 3
        if self.nir_training:
            in_channel += 1
        if self.use_pos:
            in_channel += 2
        input_dim = in_channel
        layers = []
        for output_dim in output_dims:
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1))
            # layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Conv2d(input_dim, 3, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, sample):
        if self.nir_training:
            if self.use_pos:
                # o0 = torch.cat((sample["rgb"] - sample["bg"], sample["pe"]), dim=1)
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir']- sample['bg_nir'],
                                sample["position"]), dim=1)
            else:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir'] - sample['bg_nir']), dim=1)
        else:
            if self.use_pos:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"], sample["position"]), dim=1)
            else:
                o0 = sample["rgb"] - sample["bg_rgb"]
        o = self.layers(o0)

        return o

class Model_quan_l4_16(nn.Module):
    def __init__(self, use_pos=True, dropout=0.4, output_dims=None, nir_training=True):
        super(Model_quan_l4_16, self).__init__()
        if output_dims is None:
            output_dims = [16, 16, 16]
        self.use_pos = use_pos
        self.nir_training = nir_training
        in_channel = 3
        if self.nir_training:
            in_channel += 1
        if self.use_pos:
            in_channel += 2
        input_dim = in_channel
        layers = []
        for output_dim in output_dims:
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1))
            # layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Conv2d(input_dim, 3, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, sample):
        if self.nir_training:
            if self.use_pos:
                # o0 = torch.cat((sample["rgb"] - sample["bg"], sample["pe"]), dim=1)
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir']- sample['bg_nir'],
                                sample["position"]), dim=1)
            else:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir'] - sample['bg_nir']), dim=1)
        else:
            if self.use_pos:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"], sample["position"]), dim=1)
            else:
                o0 = sample["rgb"] - sample["bg_rgb"]
        o = self.layers(o0)

        return o



class Model_quan_l3_32(nn.Module):
    def __init__(self, use_pos=True, dropout=0.4, output_dims=None, nir_training=True):
        super(Model_quan_l3_32, self).__init__()
        if output_dims is None:
            output_dims = [32, 32]
        self.use_pos = use_pos
        self.nir_training = nir_training
        in_channel = 3
        if self.nir_training:
            in_channel += 1
        if self.use_pos:
            in_channel += 2
        input_dim = in_channel
        layers = []
        for output_dim in output_dims:
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1))
            # layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Conv2d(input_dim, 3, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, sample):
        if self.nir_training:
            if self.use_pos:
                # o0 = torch.cat((sample["rgb"] - sample["bg"], sample["pe"]), dim=1)
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir']- sample['bg_nir'],
                                sample["position"]), dim=1)
            else:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir'] - sample['bg_nir']), dim=1)
        else:
            if self.use_pos:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"], sample["position"]), dim=1)
            else:
                o0 = sample["rgb"] - sample["bg_rgb"]
        o = self.layers(o0)

        return o

class Model_quan_l3_16(nn.Module):
    def __init__(self, use_pos=True, dropout=0.4, output_dims=None, nir_training=True):
        super(Model_quan_l3_16, self).__init__()
        if output_dims is None:
            output_dims = [16, 16]
        self.use_pos = use_pos
        self.nir_training = nir_training
        in_channel = 3
        if self.nir_training:
            in_channel += 1
        if self.use_pos:
            in_channel += 2
        input_dim = in_channel
        layers = []
        for output_dim in output_dims:
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1))
            # layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Conv2d(input_dim, 3, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, sample):
        if self.nir_training:
            if self.use_pos:
                # o0 = torch.cat((sample["rgb"] - sample["bg"], sample["pe"]), dim=1)
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir']- sample['bg_nir'],
                                sample["position"]), dim=1)
            else:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"],
                                sample['nir'] - sample['bg_nir']), dim=1)
        else:
            if self.use_pos:
                o0 = torch.cat((sample["rgb"] - sample["bg_rgb"], sample["position"]), dim=1)
            else:
                o0 = sample["rgb"] - sample["bg_rgb"]
        o = self.layers(o0)

        return o



class Modelv3(nn.Module):  # 缩小后的模型
    def __init__(self, use_pos=True, dropout=0.4, output_dims=None,nir_training=True, version=0, save_input=False):
        super(Modelv3, self).__init__()
        if output_dims is None:
            output_dims = [32, 32, 32]
            # output_dims = [512, 512, 512, 256]
        self.use_pos = use_pos
        self.nir_training = nir_training
        self.version = version
        in_channel = 3
        if self.nir_training:
            in_channel+=3
        if self.use_pos:
            # in_channel += 32
            in_channel += 3
        input_dim = in_channel
        layers = []
        for output_dim in output_dims:
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=1))
            # layers.append(nn.BatchNorm2d(output_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        layers.append(nn.Conv2d(input_dim, 3, kernel_size=1))
        self.layers = nn.Sequential(*layers)

        # 调试开关：是否保存输入数据
        self.save_input = save_input
        self.save_dir = f"logs/gel_experiment/version_{self.version}/"

    def save_input_to_txt(self, tensor, f):
        """
        将输入张量写入一个已打开的文件对象。
        :param tensor: 输入张量 (shape: [C, H, W])
        :param f: 已打开的文件对象
        """
        # 移除 batch 维度并转为 numpy 数组
        input_np = tensor.squeeze(0).detach().cpu().numpy()

        # 展平数据
        flat_data = input_np.flatten()

        # 构造 header 注释
        C, H, W = input_np.shape
        header = f"Input Tensor\nShape: {input_np.shape}, Dtype: {input_np.dtype}\n"

        # 将数据写入提供的文件对象
        np.savetxt(
            f,
            flat_data,
            fmt='%.6f',
            header=header,
            comments=''
        )


    def forward(self, sample):
        if self.use_pos:
            # o0 = torch.cat((sample["rgb"] - sample["bg"], sample["pe"]), dim=1)
            # 考虑背景使模型能够区分出图像中的信号哪些来自物体接触，哪些是传感器固有的，从而基于接触信息进行预测
            # 通过预先计算前景和背景的差分，模型不需要学习如何从原始图像中分离出接触和背景，而是专注于学习接触和法向之间的关系
            o0 = torch.cat((sample["rgb"] - sample["bg"], sample["position"]), dim=1)
        else:
            o0 = sample["rgb"] - sample["bg"]

        # 如果启用调试模式，保存输入数据
        if self.save_input:
            # 确保保存目录存在
            os.makedirs(self.save_dir, exist_ok=True)
            filename = os.path.join(self.save_dir, "input.txt")
            # 先创建（打开）文件，然后将文件对象传递给写入函数
            with open(filename, 'w') as f:
                self.save_input_to_txt(o0, f)
        o = self.layers(o0)

        return o

class ForceNet(nn.Module):
    def __init__(self, use_pos: bool = False, hidden: int = 8, out_channels: int = 1):
        super(ForceNet, self).__init__()
        self.use_pos = use_pos
        if self.use_pos:
            # dep_pos: [C,H,W]，我们现在约定 C>=5: [dz, dx, dy, posX, posY]
            # 这里简单起见，直接把 dep_pos 的所有通道作为输入
            in_channels = None  # 运行时根据 sample['dep_pos'].shape[1] 推断
        else:
            in_channels = 1

        self.hidden = hidden
        self.out_channels = out_channels
        # 占位，真正的 Conv 会在 forward 里 lazy-init
        self.net = None

    def _build_net(self, in_channels: int):
        # we avoid builtin zero-padding in Conv2d and instead apply replication padding manually
        # to keep behavior consistent between training and inference and to prevent
        # artificial edge jumps caused by zero padding.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden, kernel_size=5, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden, self.out_channels, kernel_size=1, bias=True),
        )

    def forward(self, sample):
        if self.use_pos:
            x = sample['dep_pos']  # [B, C, H, W]，C>=5=[dz,dx,dy,posX,posY]
        else:
            x = sample['dep']      # [B, 1, H, W]

        if self.net is None:
            in_c = x.shape[1]
            self._build_net(in_c)
        # apply replication padding to match original receptive field (kernel_size=5 -> pad=2)
        pad = 2
        if pad > 0:
            # F.pad expects pad format (left, right, top, bottom)
            x = F.pad(x, (pad, pad, pad, pad), mode='replicate')

        return self.net(x)


# class WeightGenerator(nn.Module):
#     """
#     位置网络 (HyperNetwork):
#     输入: 坐标 (X, Y, Z) [B, 3, H, W]
#     输出: 该位置对应的卷积核权重 [B, kernel_size^2, H, W]
#     物理含义: 预测每个空间位置的局部刚度矩阵 k(x)
#     """
#
#     def __init__(self, in_channels=3, kernel_size=5, hidden_dim=32):
#         super().__init__()
#         self.kernel_size = kernel_size
#         out_channels = kernel_size * kernel_size  # 输出 25 个权重值对应 5x5 核
#
#         # 使用 1x1 卷积相当于在每个像素位置独立运行一个 MLP
#         # 这样可以保持空间分辨率，同时学习空间变化的函数
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
#             # 最后一层不加激活函数，因为刚度/权重可以是负的(虽然物理上刚度通常为正，但网络权重需要灵活)
#         )
#
#     def forward(self, pos):
#         return self.net(pos)


class WeightGenerator(nn.Module):
    """
    位置网络 (HyperNetwork):
    输入: 坐标 (X, Y, Z) [B, 3, H, W]
    输出: 动态卷积核权重
    """

    def __init__(self, in_channels=3, kernel_size=5, input_feat_dim=3, hidden_dim=64):
        """
        input_feat_dim: 输入位移场的通道数 (dx, dy, dz 共 3 个)
        """
        super().__init__()
        self.kernel_size = kernel_size
        # 输出通道数 = 输入特征通道数 * 卷积核面积
        # 例如: 3 * 5 * 5 = 75
        out_channels = input_feat_dim * kernel_size * kernel_size

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        )

    def forward(self, pos):
        return self.net(pos)


class HyperForceNet(nn.Module):
    """
    全耦合动态卷积力估计网络
    Input: (dx, dy, dz) -> Output: (Fx, Fy, Fz)
    """

    def __init__(self, kernel_size=5, stride=4, pos_dim=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = stride

        # <--- 2. 必须加入下采样层
        if self.stride > 1:
            self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # 输入特征维度是 3 (dx, dy, dz)
        input_feat_dim = 3

        # 我们需要三个独立的生成器，分别生成用于预测 Fx, Fy, Fz 的卷积核
        # 每个生成器都会产生一个 [3, 5, 5] 的卷积核（展平后为 75）
        # 这意味着 Fx = w_xx*dx + w_xy*dy + w_xz*dz + bias
        self.weight_gen_x = WeightGenerator(in_channels=pos_dim, kernel_size=kernel_size, input_feat_dim=input_feat_dim)
        self.weight_gen_y = WeightGenerator(in_channels=pos_dim, kernel_size=kernel_size, input_feat_dim=input_feat_dim)
        self.weight_gen_z = WeightGenerator(in_channels=pos_dim, kernel_size=kernel_size, input_feat_dim=input_feat_dim)

        # 偏置生成器 (Bias)
        self.bias_gen = nn.Sequential(
            nn.Conv2d(pos_dim, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)  # 输出 3 通道偏置 (bx, by, bz)
        )

    def _unfold_displacement(self, displacement):
        B, C, H, W = displacement.shape
        K = self.kernel_size

        if self.padding > 0:
            displacement_padded = F.pad(displacement, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
            disp_unfolded = F.unfold(displacement_padded, kernel_size=K, padding=0)
        else:
            disp_unfolded = F.unfold(displacement, kernel_size=K, padding=0)

        return disp_unfolded, B, C, H, W, K

    def _apply_dynamic_conv(self, disp_unfolded, dynamic_weights, B, C, H, W, K):
        """
        disp_unfolded: [B, 3*K*K, H*W]
        dynamic_weights: [B, 75, H, W] (3*K*K)
        return: [B, 1, H, W] (Single force component map)
        """
        # 调整权重形状
        # [B, 75, H, W] -> [B, 75, H*W]
        weights_flat = dynamic_weights.view(B, C * K * K, -1)

        # 动态卷积 (点积求和)
        # out: [B, 1, H*W]
        out_flat = (disp_unfolded * weights_flat).sum(dim=1, keepdim=True)

        # 恢复形状 [B, 1, H, W]
        return out_flat.view(B, 1, H, W)

    def dynamic_forward(self, disp_cat, pos):
        """
        disp_cat: [B, 3, H, W] -> 包含 dx, dy, dz
        pos: [B, 3, H, W] -> 包含 X, Y, Z 坐标
        """
        # 在计算前进行下采样
        if self.stride > 1:
            disp_cat = self.downsample(disp_cat)
            pos = self.downsample(pos)

        # 1. 生成针对 Fx, Fy, Fz 的三组动态权重
        # 每组权重大小为 [B, 75, H, W]，不仅看当前通道，也看其他两个通道的位移
        w_fx = self.weight_gen_x(pos)
        w_fy = self.weight_gen_y(pos)
        w_fz = self.weight_gen_z(pos)

        # 2. 生成偏置 [B, 3, H, W]
        bias = self.bias_gen(pos)

        disp_unfolded, B, C, H, W, K = self._unfold_displacement(disp_cat)

        # 3. 分别计算三个力分量 (全耦合计算)
        # Fx 由 (dx, dy, dz) 共同决定
        fx_map = self._apply_dynamic_conv(disp_unfolded, w_fx, B, C, H, W, K)
        fy_map = self._apply_dynamic_conv(disp_unfolded, w_fy, B, C, H, W, K)
        fz_map = self._apply_dynamic_conv(disp_unfolded, w_fz, B, C, H, W, K)

        # 4. 拼接结果 [B, 3, H, W]
        force_map = torch.cat([fx_map, fy_map, fz_map], dim=1)

        # 5. 加上偏置
        return force_map + bias

    def forward(self, sample):
        # 1. 准备全耦合输入
        dep_z = sample['dz']  # [B, 1, H, W]
        dep_x = sample.get('dx', None)  # [B, 1, H, W]
        dep_y = sample.get('dy', None)  # [B, 1, H, W]
        pos = sample['pos']  # [B, 3, H, W]

        # 如果 Dataset 没有提供 dx/dy (比如还没算出来)，就用 0 填充
        if dep_x is None: dep_x = torch.zeros_like(dep_z)
        if dep_y is None: dep_y = torch.zeros_like(dep_z)

        # 拼接: [B, 3, H, W]
        disp_real = torch.cat([dep_x, dep_y, dep_z], dim=1)

        # 2. 计算真实预测
        pred = self.dynamic_forward(disp_real, pos)

        # 3. 零点校准 (Zero Calibration)
        # 背景输入的位移全是 0
        disp_bg = torch.zeros_like(disp_real)
        pred_bg = self.dynamic_forward(disp_bg, pos)

        # 4. 缩放与输出
        scaling = 1000.0
        # scaling = 1.0
        return (pred - pred_bg) / scaling
