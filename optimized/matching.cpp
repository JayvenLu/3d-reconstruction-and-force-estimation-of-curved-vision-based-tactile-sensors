// Port of the local find_marker.py objective and DFS branches.
// Double precision, no fast-math. Candidates are replayed through the SAME branches.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
namespace py = pybind11;
using Clock = std::chrono::steady_clock;
struct P { double x=0,y=0; P operator+(P b)const{return{x+b.x,y+b.y};}
 P operator-(P b)const{return{x-b.x,y-b.y};} P operator/(double v)const{return{x/v,y/v};} };
double sq(P p){return p.x*p.x+p.y*p.y;}
struct Move {int r,c,gap,k; bool start,skip;};
struct Solver {
 int N,M,n; double dx,dy,dmin,dmax,moving,flow_limit,threshold,best=-1;
 std::vector<P> O,C,D,bestD;
 std::vector<int> row,col,occ,done,bestOcc,bestRow,bestCol,seed;
 std::vector<double> first;
 std::vector<unsigned char> pred;
 bool timeout=false,seedAccepted=false,seedMode=false,abortSeed=false;
 long long nodes=0;
 Clock::time_point deadline;
 Solver(int nr,int nc,double x,double y):N(nr),M(nc),dx(x),dy(y){
  dmin=std::pow(dx*.5,2);dmax=std::pow(dx*1.8,2);moving=dx*2;flow_limit=dx*.8;
  threshold=15000*std::pow(dx/21,2);
 }
 int at(int r,int c) const{return r*M+c;}
 bool expired(){if(seedMode)return false;if(timeout)return true;
  if(Clock::now()>=deadline){timeout=true;return true;}return false;}
 void clear(){row.assign(n,-1);col.assign(n,-1);occ.assign(N*M,-1);done.assign(N,0);first.assign(N,0);}
 double cost(int i){int r=row[i],c=col[i];P f=C[i]-O[at(r,c)];double v=.1*sq(f);
  const int dr[]={0,-1,1},dc[]={-1,0,0};
  for(int k=0;k<3;k++){int a=r+dr[k],b=c+dc[k];if(a<0||a>=N||b<0||b>=M)continue;
   int j=occ[at(a,b)];if(j>=0){double d=sq(C[j]-O[at(a,b)]-f);v+=std::sqrt(d)>=flow_limit?1e8:d;}}
  return v;
 }
 double infer(){double v=0;int border=0;int dr[]={0,-1,0,1},dc[]={-1,0,1,0};
  for(int r=0;r<N;r++)for(int c=0;c<M;c++)if(occ[at(r,c)]<0){P sum{};int cnt=0;
   for(int k=0;k<4;k++){int a=r+dr[k],b=c+dc[k],aa=a+dr[k],bb=b+dc[k];
    if(aa>=0&&aa<N&&bb>=0&&bb<M&&occ[at(a,b)]>=0&&occ[at(aa,bb)]>=0){
     P f=C[occ[at(a,b)]]-O[at(a,b)],g=C[occ[at(aa,bb)]]-O[at(aa,bb)];sum=sum+f+(f-g);cnt++;}}
   for(int rad=1;rad<=2&&cnt==0;rad++)for(int a=r-rad;a<=r+rad;a++)for(int b=c-rad;b<=c+rad;b++)
    if(a>=0&&a<N&&b>=0&&b<M&&occ[at(a,b)]>=0){sum=sum+(C[occ[at(a,b)]]-O[at(a,b)]);cnt++;}
   D[at(r,c)]=O[at(r,c)]+(cnt?sum/(cnt+1e-6):sum);
   if(c==0&&D[at(r,c)].y>=O[at(r,c)].y-dy/2)border++;
   // Preserve the reference's N (not M) comparison for rectangular-grid parity.
   if(c==N-1&&D[at(r,c)].y<=O[at(r,c)].y+dy/2)border++;
   v+=.1*sq(D[at(r,c)]-O[at(r,c)]);
  }
  if(border>=N-1)v+=1e7;
  for(int r=0;r<N;r++)for(int c=0;c<M;c++)if(occ[at(r,c)]<0){P f=D[at(r,c)]-O[at(r,c)];
   for(int k=0;k<4;k++){int a=r+dr[k],b=c+dc[k];if(a<0||a>=N||b<0||b>=M)continue;
    if(occ[at(a,b)]>=0)v+=sq(C[occ[at(a,b)]]-O[at(a,b)]-f);
    else if(k<2)v+=sq(D[at(a,b)]-O[at(a,b)]-f);}}
  return v;
 }
 std::vector<Move> moves(int i,int missing,int spare){std::vector<Move> out;
  for(int j=0;j<i;j++)if(row[j]>=0&&pred[i*n+j]){int r=row[j],c=col[j]+1;
   if(c>=M||occ[at(r,c)]>=0)continue;bool valid=true;
   if(r>0&&occ[at(r-1,c)]>=0&&C[i].y<=C[occ[at(r-1,c)]].y)valid=false;
   if(r<N-1&&occ[at(r+1,c)]>=0&&C[i].y>=C[occ[at(r+1,c)]].y)valid=false;
   for(int k=0;k<N;k++){int s=occ[at(k,c)];if(s>=0&&((k<r&&C[s].y>C[i].y)||(k>r&&C[s].y<C[i].y)))valid=false;}
   if(valid)out.push_back({r,c,0,c-1,false,false});
  }
  for(int r=0;r<N;r++)if(!done[r]&&occ[at(r,0)]<0){bool valid=true;
   for(int k=0;k<N;k++)if(done[k]&&((k<r&&first[k]>C[i].y)||(k>r&&first[k]<C[i].y)))valid=false;
   if(valid)out.push_back({r,0,0,-1,true,false});
  }
  for(int gap=1;gap<=missing;gap++)for(int r=0;r<N;r++){
   if(std::abs(C[i].y-O[at(r,0)].y)>moving)continue;
   int k=M-1;while(k>=0&&occ[at(r,k)]==-1)k--;
   int c=k+gap+1;if(c>=M||std::sqrt(sq(C[i]-O[at(r,c)]))>moving)continue;
   out.push_back({r,c,gap,k,false,false});
  }
  if(spare>0)out.push_back({-1,-1,0,-1,false,true});
  return out;
 }
 void dfs(int i,double total,int missing,int spare){nodes++;
  if(abortSeed||expired()||(best>=0&&total>=best)||total>=threshold)return;
  if(i==n){double value=total+infer();
   if((best<0||value<best)&&value<threshold){
    for(int j=0;j<n;j++)if(row[j]>=0)D[at(row[j],col[j])]=C[j];
    best=value;bestD=D;bestOcc=occ;bestRow=row;bestCol=col;
    if(seedMode){seedAccepted=true;abortSeed=true;}
   }return;
  }
  auto opts=moves(i,missing,spare);
  int target=seed.empty()?-2:seed[i];
  if(!seedMode&&target!=-2)std::stable_partition(opts.begin(),opts.end(),[&](const Move&a){return(a.skip?-1:at(a.r,a.c))==target;});
  for(auto a:opts){if(abortSeed||expired())break;
   if(seedMode&&(a.skip?-1:at(a.r,a.c))!=target)continue;
   if(a.skip){row[i]=col[i]=-1;dfs(i+1,total,missing,spare-1);continue;}
   row[i]=a.r;col[i]=a.c;
   double oldFirst=first[a.r];int oldDone=done[a.r];
   int oldOcc=occ[at(a.r,a.c)];
   if(a.start){done[a.r]=1;first[a.r]=C[i].y;}
   for(int t=1;t<=a.gap;t++)occ[at(a.r,a.k+t)]=-2;
   occ[at(a.r,a.c)]=i;
   dfs(i+1,total+cost(i),missing-a.gap,spare);
   occ[at(a.r,a.c)]=oldOcc;
   for(int t=1;t<=a.gap;t++)occ[at(a.r,a.k+t)]=-1;
   done[a.r]=oldDone;first[a.r]=oldFirst;
  }
 }
 void run(const std::vector<std::vector<int>>& seeds,double ms,bool search){
  deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(ms));
  D.assign(N*M,{});pred.assign(n*n,0);
  for(int i=0;i<n;i++)for(int j=0;j<i;j++){double d=sq(C[i]-C[j]);
   double angle=d==0?0:std::asin(std::abs(C[i].y-C[j].y)/std::sqrt(d))*180/3.14159265;
   pred[i*n+j]=angle<=70&&d>=dmin&&d<=dmax;}
  int missing=std::max(0,N*M-n),spare=std::max(0,n-N*M);
  for(auto candidate:seeds){seed=candidate;clear();seedMode=true;abortSeed=false;
   int skipped=int(std::count(seed.begin(),seed.end(),-1));
   if(skipped<=spare)dfs(0,0,missing,spare);
   seedMode=false;abortSeed=false;}
  if(best>=0){seed.resize(n);for(int i=0;i<n;i++)seed[i]=bestRow[i]<0?-1:at(bestRow[i],bestCol[i]);}
  else seed.clear();
  if(search){clear();dfs(0,0,missing,spare);}
  // Match the reference retry policy: relax missing/spare ONLY if the first
  // problem has no current-frame solution. Otherwise dropping a genuine marker
  // could lower the objective but silently change a perfectly valid association.
  if(best<0){
   for(auto candidate:seeds){seed=candidate;clear();seedMode=true;abortSeed=false;
    int skipped=int(std::count(seed.begin(),seed.end(),-1));
    if(skipped<=spare+1)dfs(0,0,missing+1,spare+1);
    seedMode=false;abortSeed=false;}
   if(search&&!timeout){clear();dfs(0,0,missing+1,spare+1);}
  }
 }
};
py::dict solve(py::array_t<double,py::array::c_style|py::array::forcecast> ref,
 py::array_t<double,py::array::c_style|py::array::forcecast> points,
 py::array_t<int,py::array::c_style|py::array::forcecast> candidates,
 int rows,int cols,double dx,double dy,double budget_ms,bool search){
 if(rows<1||cols<1||rows>30||cols>30||ref.ndim()!=2||ref.shape(0)!=rows*cols||ref.shape(1)!=2||
  points.ndim()!=2||points.shape(1)!=2||points.shape(0)>900||candidates.ndim()!=2||candidates.shape(1)!=points.shape(0))throw std::invalid_argument("Invalid matching shapes");
 auto a=ref.unchecked<2>();auto b=points.unchecked<2>();auto s=candidates.unchecked<2>();
 Solver solver(rows,cols,dx,dy);solver.n=int(points.shape(0));
 for(int i=0;i<rows*cols;i++)solver.O.push_back({a(i,0),a(i,1)});
 for(int i=0;i<solver.n;i++)solver.C.push_back({b(i,0),b(i,1)});
 std::vector<std::vector<int>> seeds;
 for(int j=0;j<candidates.shape(0);j++){std::vector<int> v;for(int i=0;i<solver.n;i++)v.push_back(s(j,i));seeds.push_back(v);}
 auto start=Clock::now();
 {py::gil_scoped_release release;solver.run(seeds,budget_ms,search);}
 py::array_t<double> current({rows*cols,2});py::array_t<int> occupied(rows*cols);
 auto o=current.mutable_unchecked<2>();auto q=occupied.mutable_unchecked<1>();
 for(int i=0;i<rows*cols;i++){o(i,0)=solver.best<0?NAN:solver.bestD[i].x;o(i,1)=solver.best<0?NAN:solver.bestD[i].y;q(i)=solver.best<0?-1:solver.bestOcc[i];}
 py::dict result;result["current"]=current;result["occupied"]=occupied;result["cost"]=solver.best;
 result["valid"]=solver.best>=0;result["timed_out"]=solver.timeout;result["seed_accepted"]=solver.seedAccepted;result["nodes"]=solver.nodes;
 result["elapsed_ms"]=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return result;
}
PYBIND11_MODULE(_matching,m){m.def("solve",&solve,py::arg("reference"),py::arg("points"),py::arg("candidates"),py::arg("rows"),py::arg("cols"),py::arg("dx"),py::arg("dy"),py::arg("budget_ms")=8.0,py::arg("search")=true);}
