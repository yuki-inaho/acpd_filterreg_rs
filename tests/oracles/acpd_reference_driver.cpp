// Test harness only: algorithms are in the independently extracted original header.
#include "acpd_reference.hpp"
#include <iostream>
#include <iomanip>
MatrixXd read(int n,int d) {MatrixXd a(n,d);for(int i=0;i<n;++i)for(int j=0;j<d;++j)std::cin>>a(i,j);return a;}
void mat(const MatrixXd& a){std::cout<<'[';for(int i=0;i<a.rows();++i){if(i)std::cout<<',';std::cout<<'[';
for(int j=0;j<a.cols();++j){if(j)std::cout<<',';std::cout<<a(i,j);}std::cout<<']';}std::cout<<']';}
void vec(const VectorXd& a){std::cout<<'[';for(int i=0;i<a.size();++i){if(i)std::cout<<',';std::cout<<a[i];}std::cout<<']';}
int main(){std::cout<<std::setprecision(17);try{
std::string op;std::cin>>op;
if(op=="stats") {int d,n,m;double sigma,w;std::cin>>d>>n>>m>>sigma>>w;auto x=read(n,d),y=read(m,d);
auto s=ComputePosteriorP(x,y,sigma,w);VectorXd x2=s.P*x.rowwise().squaredNorm();
std::cout<<"{\"rho\":";vec(s.rowSums);std::cout<<",\"px\":";mat(s.PX);std::cout<<",\"x2\":";vec(x2);
std::cout<<",\"mass\":"<<s.NP<<",\"initial_sigma2\":"<<InitializeSigma2(x,y)<<",\"sigma2\":"
<<UpdateSigma2FromStats(x,y,s.rowSums,s.colSums,s.PX)<<'}';
} else if(op=="schedule") {int t,d;std::cin>>t>>d;std::cout<<'[';for(int i=0;i<t;++i){if(i)std::cout<<',';std::cout<<DegreeScheduleDecreasingStages(i,t,d);}std::cout<<']';
} else if(op=="fit") {int d,n,degree;std::cin>>d>>n>>degree;auto y=read(n,d),z=read(n,d);VectorXd w=read(n,1),center=VectorXd::Zero(d);MatrixXd out=y;
if(d==2)ref2d::AMVFF2D_MStep(out,z,y,w,center,degree);else ref3d::AMVFF3D_MStep(out,y,z,w,center,degree);mat(out);
} else if(op=="basis") {int d,n,q;std::cin>>d>>n>>q;MatrixXd y=read(n,d),out(n,TotalMonomialCount(d,q));
if(d==2){MatrixXd b;VectorXd zero=VectorXd::Zero(2);ref2d::GetMatB4Analytic(b,q,zero,y);
int col=0,offset=0;for(int r=0;r<=q;++r){for(int k=0;k<=r;++k){for(int i=0;i<n;++i)out(i,col)=b(2*i,offset+k);++col;}offset+=2*(r+1);}}
else{VectorXd zero=VectorXd::Zero(3);for(int i=0;i<n;++i){VectorXd p=y.row(i).transpose();int off=0;for(int r=0;r<=q;++r){auto c=GetTaylorCoef3D(p,zero,r);out.row(i).segment(off,c.size())=c.transpose();off+=c.size();}}}mat(out);
}else throw std::runtime_error("unknown oracle command");
std::cout<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 3;}}
