// Reusable ablation and profiling entry point for the C++ core.
//
// Wall-clock seconds on a shared machine are indicative only. Every run also
// reports load-independent work counters, and `tools/benchmark_speed.py --ir`
// wraps this binary in callgrind for a deterministic instruction count. Prefer
// those two when comparing builds.
#include "acpd/analytic.hpp"
#include "acpd/rigid.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace acpd;
namespace {
    // Deterministic per (seed,n,d): a smooth surface plus a known non-rigid warp,
    // so a run is reproducible and comparable across builds.
    Matrix cloud(int n,int d,unsigned seed) {
        std::mt19937 engine(seed);
        std::uniform_real_distribution<double> uniform(-1,1);
        Matrix points(n,d);
        for(int i=0;i<n;++i) for(int a=0;a<d;++a) points(i,a)=uniform(engine)*(a+1);
        return points;
    }
    Matrix warp(const Matrix& moving) {
        Matrix out=moving;
        for(int i=0;i<out.rows();++i) out(i,0)+=0.06+0.03*moving(i,1)*moving(i,1);
        return out;
    }
    std::string argument(int argc,char** argv,const char* name,const std::string& fallback) {
        for(int i=1;i+1<argc;++i) if(std::strcmp(argv[i],name)==0) return argv[i+1];
        return fallback;
    }
    int integer(int argc,char** argv,const char* name,int fallback) {
        const std::string value=argument(argc,argv,name,"");
        return value.empty()?fallback:std::stoi(value);
    }
}
int main(int argc,char** argv) {
    const int n=integer(argc,argv,"--n",500), d=integer(argc,argv,"--d",3);
    const int reps=integer(argc,argv,"--reps",1);
    const unsigned seed=static_cast<unsigned>(integer(argc,argv,"--seed",7));
    const std::string stage=argument(argc,argv,"--stage","nonrigid");
    const std::string backend=argument(argc,argv,"--backend","permutohedral");
    // "deformed" is the realistic workload: a known smooth warp of the same cloud.
    // "independent" is deliberately ill-posed and exercises the divergence path.
    const std::string pairing=argument(argc,argv,"--pair","deformed");
    const Matrix moving=cloud(n,d,seed);
    const Matrix fixed=warp(pairing=="independent"?cloud(n,d,seed+1):moving);
    Options options;
    options.backend=backend_from_string(backend);
    options.rigid.sigma2=0.08;
    options.fgt.order=integer(argc,argv,"--fgt-order",options.fgt.order);
    options.fgt.max_clusters=integer(argc,argv,"--fgt-max-clusters",options.fgt.max_clusters);
    options.fgt.cluster_radius=std::stod(argument(argc,argv,"--fgt-cluster-radius",
    std::to_string(options.fgt.cluster_radius)));
    options.fgt.cutoff_radius=std::stod(argument(argc,argv,"--fgt-cutoff-radius",
    std::to_string(options.fgt.cutoff_radius)));
    options.cuda.single_precision=integer(argc,argv,"--cuda-single-precision",0)!=0;
    // ACPD E-step backend, independent of the FilterReg backend above.
    options.analytic.backend=backend_from_string(argument(argc,argv,"--analytic-backend","direct"));
    if(stage=="estep") {
        // Isolates the Gaussian transform: no M-step, no stopping logic.
        const double sigma2=std::stod(argument(argc,argv,"--sigma2","0.05"));
        const auto begin=std::chrono::steady_clock::now();
        double mass=0;
        int vertices=0;
        for(int r=0;r<reps;++r) {
            const Statistics statistics=posterior_statistics(fixed,moving,sigma2,0.1,true,options.backend,
            Matrix(),nullptr,options.fgt,options.cuda);
            mass+=statistics.mass;
            vertices=statistics.vertices;
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        std::cout<<"{\"stage\":\"estep\",\"backend\":\""<<backend<<"\",\"n\":"<<n<<",\"d\":"<<d
        <<",\"reps\":"<<reps<<",\"seconds\":"<<seconds<<",\"lattice_vertices\":"<<vertices
        <<",\"sigma2\":"<<sigma2
        <<",\"cuda_single_precision\":"<<(options.cuda.single_precision?1:0)
        <<",\"gaussian_pairs\":"<<(backend=="direct"?static_cast<long long>(n)*n:0LL)
        <<",\"checksum\":"<<mass<<"}\n";
        return 0;
    }
    options.method=method_from_string(stage);
    const int budget=integer(argc,argv,"--analytic-iterations",options.analytic.max_iterations);
    options.analytic.max_iterations=budget;
    options.analytic.max_degree=integer(argc,argv,"--max-degree",options.analytic.max_degree);
    const std::string initialization=argument(argc,argv,"--initialization","");
    if(!initialization.empty()) options.analytic.initialization=initialization;
    const auto begin=std::chrono::steady_clock::now();
    Result result=registration(fixed,moving,options,Matrix::Identity(d,d),Vector::Zero(d));
    for(int r=1;r<reps;++r) result=registration(fixed,moving,options,Matrix::Identity(d,d),Vector::Zero(d));
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    double error=0;
    for(int i=0;i<fixed.rows();++i) error+=(result.transformed.row(i)-fixed.row(i)).squaredNorm();
    int degree=0;
    long long design_work=0;
    for(const auto& item:result.analytic_stage.history) {
        degree=std::max(degree,item.degree);
        // Rows x basis^2: the dominant M-step cost of a dense least-squares solve.
        const long long basis=static_cast<long long>(exponents(d,item.degree).size());
        design_work+=static_cast<long long>(item.active)*basis*basis;
    }
    std::cout<<"{\"stage\":\""<<stage<<"\",\"backend\":\""<<backend<<"\",\"n\":"<<n<<",\"d\":"<<d
    <<",\"reps\":"<<reps<<",\"seconds\":"<<seconds
    <<",\"rigid_iterations\":"<<result.rigid_stage.history.size()
    <<",\"analytic_iterations\":"<<result.analytic_stage.history.size()
    <<",\"index_builds\":"<<result.rigid_stage.index_builds
    <<",\"max_degree_reached\":"<<degree
    <<",\"mstep_row_basis_squared\":"<<design_work
    <<",\"pair\":\""<<pairing<<"\""
    <<",\"analytic_backend\":\""<<name(options.analytic.backend)<<"\""
    <<",\"stop_reason\":\""<<result.analytic_stage.stop_reason<<"\""
    <<",\"paired_rms\":"<<std::sqrt(error/fixed.rows())<<"}\n";
    return 0;
}
