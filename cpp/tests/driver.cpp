// Test-only textual bridge. Never imported or selected by the public package.
#include "acpd/analytic.hpp"
#include "acpd/rigid.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
using namespace acpd;
namespace {
Matrix read_matrix(int n,int d) {
    if(n<0||d<0||n>1000000||d>1000) throw std::invalid_argument("invalid test input shape");
    Matrix a(n,d);
    for(int i=0;i<n;++i) for(int j=0;j<d;++j)
        if(!(std::cin>>a(i,j))) throw std::invalid_argument("truncated or invalid driver input");
    return a;
}
void scalar(double a) {if(std::isfinite(a)) std::cout<<a; else std::cout<<"null";}
void mat(const Matrix& a) {
    std::cout<<'[';
    for(int i=0;i<a.rows();++i) {if(i)std::cout<<',';std::cout<<'[';
        for(int j=0;j<a.cols();++j){if(j)std::cout<<',';scalar(a(i,j));}std::cout<<']';}
    std::cout<<']';
}
void vec(const Vector& a) {
    std::cout<<'[';for(int i=0;i<a.size();++i){if(i)std::cout<<',';scalar(a[i]);}std::cout<<']';
}
void stage(const Stage& s) {
    std::cout<<"{\"initial_sigma2\":"<<s.initial_sigma2<<",\"final_sigma2\":"<<s.final_sigma2
        <<",\"converged\":"<<(s.converged?"true":"false")<<",\"stop_reason\":\""<<s.stop_reason
        <<"\",\"best_iteration\":"<<s.best_iteration<<",\"index_builds\":"<<s.index_builds<<",\"history\":[";
    for(std::size_t i=0;i<s.history.size();++i) {
        if(i)std::cout<<',';const auto& h=s.history[i];
        std::cout<<"{\"iteration\":"<<h.iteration<<",\"raw_degree\":"<<h.raw_degree<<",\"degree\":"<<h.degree
            <<",\"active\":"<<h.active<<",\"rank\":"<<h.rank<<",\"sigma2\":"<<h.sigma2<<",\"nll_before\":";
        scalar(h.nll_before);
        std::cout<<",\"step_rms\":"<<h.step_rms<<",\"fit_rms\":"<<h.fit_rms<<",\"lattice_vertices\":"<<h.lattice_vertices
            <<",\"lattice_mode\":\""<<h.lattice_mode<<"\"}";
    }
    std::cout<<"]}";
}
void result(const Result& r) {
    std::cout<<"{\"rotation\":";mat(r.rotation);std::cout<<",\"translation\":";vec(r.translation);
    std::cout<<",\"center\":";vec(r.center);std::cout<<",\"normalization_scale\":"<<r.normalization_scale;
    std::cout<<",\"transformed\":";mat(r.transformed);std::cout<<",\"rigid_transformed\":";mat(r.rigid_transformed);
    std::cout<<",\"sigma2\":"<<r.sigma2<<",\"method\":\""<<name(r.method)<<"\",\"backend\":\""<<name(r.backend)
        <<"\",\"rigid_stage\":";stage(r.rigid_stage);std::cout<<",\"analytic_stage\":";stage(r.analytic_stage);
    std::cout<<",\"steps\":[";
    for(std::size_t i=0;i<r.steps.size();++i){if(i)std::cout<<',';
        std::cout<<"{\"degree\":"<<r.steps[i].degree<<",\"coefficients\":";mat(r.steps[i].coefficients);std::cout<<'}';}
    std::cout<<"]}";
}
Statistics fit_stats(const Matrix& targets,const Vector& rho) {
    Statistics s;s.rho=rho;s.mass=rho.sum();s.px=targets;s.x2=Vector::Zero(rho.size());
    for(int i=0;i<targets.rows();++i){s.px.row(i)*=rho[i];s.x2[i]=rho[i]*targets.row(i).squaredNorm();}
    return s;
}
}
int main() {
    std::cout<<std::setprecision(17);
    try {
        std::string op;std::cin>>op;
        if(op=="registration") {
            int d,n,m,nn;std::string method,backend;Options o;
            std::cin>>d>>n>>m>>nn>>method>>backend;
            o.method=method_from_string(method);o.backend=backend_from_string(backend);
            auto& a=o.rigid;
            std::cin>>a.max_iterations>>a.tolerance>>a.w>>a.sigma2>>a.min_sigma2>>a.update_sigma2
                >>a.solver>>a.objective>>a.inner_iterations;
            auto& b=o.analytic;
            std::cin>>b.max_iterations>>b.tolerance>>b.w>>b.sigma2>>b.min_sigma2>>b.min_degree>>b.max_degree
                >>b.rank_tolerance>>b.min_mass>>b.initialization>>b.stable_patience>>b.no_improve_patience
                >>b.min_iterations>>b.improvement_relative>>b.rebound_relative>>b.divergence_radius;
            std::string analytic_backend;
            std::cin>>analytic_backend>>o.fgt.order>>o.fgt.max_clusters>>o.fgt.cluster_radius>>o.fgt.cutoff_radius>>o.cuda.single_precision;
            b.backend=backend_from_string(analytic_backend);
            Matrix x=read_matrix(n,d),y=read_matrix(m,d),r=read_matrix(d,d);
            Vector t=read_matrix(d,1);Matrix normals=read_matrix(nn,d);
            result(registration(x,y,o,r,t,normals));
        } else if(op=="stats") {
            int d,n,m,inverse;double sigma2,w;std::string backend;
            std::cin>>d>>n>>m>>sigma2>>w>>inverse>>backend;
            Matrix x=read_matrix(n,d),y=read_matrix(m,d);
            auto s=posterior_statistics(x,y,sigma2,w,inverse,backend_from_string(backend));
            std::cout<<"{\"rho\":";vec(s.rho);std::cout<<",\"px\":";mat(s.px);std::cout<<",\"x2\":";vec(s.x2);
            std::cout<<",\"mass\":"<<s.mass<<",\"nll\":";scalar(s.nll);
            std::cout<<",\"vertices\":"<<s.vertices<<",\"unsupported\":"<<s.unsupported<<",\"lattice_mode\":\""<<s.lattice_mode<<"\"}";
        } else if(op=="gaussian") {
            int d,n,m,k;double sigma2;std::string backend;FgtOptions fgt;
            CudaOptions device;
            std::cin>>d>>n>>m>>k>>sigma2>>backend
            >>fgt.order>>fgt.max_clusters>>fgt.cluster_radius>>fgt.cutoff_radius>>device.single_precision;
            Matrix s=read_matrix(n,d),q=read_matrix(m,d),v=read_matrix(n,k);
            mat(gaussian_sum(s,q,v,sigma2,backend_from_string(backend),fgt,device));
        } else if(op=="lattice") {
            int d,n,k,start;bool blur,reverse;std::cin>>d>>n>>k>>blur>>start>>reverse;
            Matrix f=read_matrix(n,d),v=read_matrix(n,k);Permutohedral lattice(f,blur);
            std::cout<<"{\"values\":";mat(lattice.filter(v,start,reverse));std::cout<<",\"vertices\":"<<lattice.lattice_size()<<'}';
        } else if(op=="simplex") {
            int d,n;bool blur;std::cin>>d>>n>>blur;Matrix f=read_matrix(n,d);std::cout<<'[';
            for(int i=0;i<n;++i){if(i)std::cout<<',';auto q=enclosing_simplex(f.row(i),blur);
                std::cout<<"{\"keys\":[";
                for(std::size_t k=0;k<q.keys.size();++k){if(k)std::cout<<',';std::cout<<'[';
                    for(int a=0;a<d;++a){if(a)std::cout<<',';std::cout<<q.keys[k][a];}std::cout<<']';}
                std::cout<<"],\"weights\":[";for(std::size_t k=0;k<q.weights.size();++k){if(k)std::cout<<',';scalar(q.weights[k]);}std::cout<<"]}";}
            std::cout<<']';
        } else if(op=="variance") {
            int d,m;double floor;std::cin>>d>>m>>floor;Matrix y=read_matrix(m,d);Statistics stats;
            stats.rho=read_matrix(m,1);stats.px=read_matrix(m,d);stats.x2=read_matrix(m,1);stats.mass=stats.rho.sum();
            scalar(variance_from_statistics(y,stats,floor));
        } else if(op=="basis") {
            int d,n,degree;std::cin>>d>>n>>degree;mat(basis(read_matrix(n,d),degree));
        } else if(op=="schedule") {
            int t,lo,hi;std::cin>>t>>lo>>hi;auto a=degree_schedule(t,lo,hi);std::cout<<'[';
            for(std::size_t i=0;i<a.size();++i){if(i)std::cout<<',';std::cout<<a[i];}std::cout<<']';
        } else if(op=="fit") {
            int d,n,degree;std::cin>>d>>n>>degree;
            Matrix y=read_matrix(n,d),z=read_matrix(n,d);Vector w=read_matrix(n,1);
            AnalyticOptions opt;auto f=fit_analytic(y,fit_stats(z,w),degree,opt);
            std::cout<<"{\"next\":";mat(f.next);std::cout<<",\"coefficients\":";mat(f.step.coefficients);
            std::cout<<",\"degree\":"<<f.step.degree<<",\"rank\":"<<f.rank<<",\"fit_rms\":"<<f.fit_rms<<'}';
        } else if(op=="rigid_fit") {
            int d,n,inner,plane;std::string solver;std::cin>>d>>n>>inner>>plane>>solver;
            Matrix y=read_matrix(n,d),z=read_matrix(n,d);Vector w=read_matrix(n,1);
            auto stats=fit_stats(z,w);if(plane) stats.normals=read_matrix(n,d);
            FilterOptions opt;opt.solver=solver;opt.objective=plane?"point_to_plane":"point_to_point";opt.inner_iterations=inner;
            auto fit=fit_rigid(y,stats,opt);
            std::cout<<"{\"rotation\":";mat(fit.rotation);std::cout<<",\"translation\":";vec(fit.translation);
            std::cout<<",\"next\":";mat(fit.next);std::cout<<",\"rank\":"<<fit.rank<<",\"fit_rms\":"<<fit.fit_rms<<'}';
        } else if(op=="twist") {
            int d;std::cin>>d;Vector p=read_matrix(d,1),delta=read_matrix(d==2?3:6,1);
            auto pose=twist_pose(delta,d);
            std::cout<<"{\"jacobian\":";mat(twist_jacobian(p));std::cout<<",\"rotation\":";mat(pose.rotation);
            std::cout<<",\"translation\":";vec(pose.translation);std::cout<<'}';
        } else throw std::invalid_argument("unknown driver operation");
        if(std::cin.fail()) throw std::invalid_argument("invalid driver input");
        std::cout<<'\n';return 0;
    } catch(const std::invalid_argument& e){std::cerr<<"ValueError: "<<e.what()<<'\n';return 2;}
      catch(const std::exception& e){std::cerr<<"RuntimeError: "<<e.what()<<'\n';return 3;}
}
