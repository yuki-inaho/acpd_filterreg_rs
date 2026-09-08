#include "filterreg_helper.hpp"
#include <iostream>
#include <iomanip>
template<int D> void evaluate(int n){std::cout<<'[';for(int i=0;i<n;++i){float f[D],w[D+2];poser::LatticeCoordKey<D> keys[D+1];
for(int a=0;a<D;++a)std::cin>>f[a];poser::permutohedral_lattice_noblur<D>(f,keys,w);
if(i)std::cout<<',';std::cout<<"{\"keys\":[";for(int k=0;k<=D;++k){if(k)std::cout<<',';std::cout<<'[';for(int a=0;a<D;++a){if(a)std::cout<<',';std::cout<<keys[k].key[a];}std::cout<<']';}
std::cout<<"],\"weights\":[";for(int k=0;k<=D;++k){if(k)std::cout<<',';std::cout<<w[k];}std::cout<<"]}";}std::cout<<']';}
int main(){int d,n;std::cin>>d>>n;std::cout<<std::setprecision(9);if(d==2)evaluate<2>(n);else if(d==3)evaluate<3>(n);else return 2;std::cout<<'\n';}
