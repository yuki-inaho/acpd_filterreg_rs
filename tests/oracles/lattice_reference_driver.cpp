#include "permutohedral.h"
#include <iostream>
#include <iomanip>
int main(){int d,n,k,blur,start,reverse;std::cin>>d>>n>>k>>blur>>start>>reverse;
Eigen::MatrixXf f(d,n),v(k,n);for(int i=0;i<n;++i)for(int a=0;a<d;++a)std::cin>>f(a,i);
for(int i=0;i<n;++i)for(int a=0;a<k;++a)std::cin>>v(a,i);
// Original compute fails to forward its start parameter. Explicit zero prefix
// gives the intended registration augmentation without altering original code.
for(int i=0;i<start;++i)v.col(i).setZero();
Permutohedral lattice;lattice.init(f,blur);auto out=lattice.compute(v,reverse,start);
std::cout<<std::setprecision(9)<<"{\"values\":[";
for(int i=0;i<n;++i){if(i)std::cout<<',';std::cout<<'[';for(int a=0;a<k;++a){if(a)std::cout<<',';std::cout<<out(a,i);}std::cout<<']';}
std::cout<<"],\"vertices\":"<<lattice.getLatticeSize()<<"}\n";}
