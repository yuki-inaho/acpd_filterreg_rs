#include "acpd/analytic.hpp"
#include "acpd/rigid.hpp"
#include <iostream>
#include <functional>
#include <cmath>
#include <limits>
using namespace acpd;
namespace {
int checks=0;
void check(bool ok,const char* label){++checks;if(!ok)throw std::runtime_error(label);}
void close(const Matrix& a,const Matrix& b,double tol,const char* label){check(a.rows()==b.rows()&&a.cols()==b.cols()&&(a-b).norm()<=tol,label);}
void throws(const std::function<void()>& f,const char* label){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,label);}
Matrix cloud(int n,int d){Matrix y(n,d);for(int i=0;i<n;++i)for(int a=0;a<d;++a)y(i,a)=std::sin((i+1)*(a+1)*1.371)+.3*std::cos((i+2)*(a+2)*.923);return y;}
Statistics paired(const Matrix& z){Statistics s;s.rho=Vector::Ones(z.rows());s.px=z;s.x2=z.rowwise().squaredNorm();s.mass=z.rows();return s;}
void dimension_tests(int d){
    const Matrix y=cloud(80,d),x=cloud(67,d)*.85;Vector point=y.row(0).transpose();
    for(bool blur:{false,true}){
        auto simplex=enclosing_simplex(y.row(0),blur);double sum=0;
        for(auto w:simplex.weights){check(w>=-1e-14,"simplex weight nonnegative");sum+=w;}
        check(std::abs(sum-1)<1e-13,"simplex partition of unity");
        Permutohedral lattice(y,blur);auto one=Matrix::Ones(y.rows(),2).eval();auto filtered=lattice.filter(one);
        check(filtered.allFinite()&&(filtered.array()>=0).all(),"lattice nonnegative");
        close(lattice.filter(2*one),2*filtered,1e-12,"lattice linearity");
        close(lattice.filter(one,static_cast<int>(y.rows())),Matrix::Zero(y.rows(),2),1e-14,"zero splat");
    }
    auto vals=moment_values(x);FixedNoBlurLattice cache(x,vals);auto count=cache.lattice_size();
    auto cached=cache.slice(y);check(cache.lattice_size()==count,"query must not mutate target index");
    close(cached,lattice_transform(x,y,vals,1,Backend::PermutohedralNoBlur).values,1e-12,"cached no-blur equality");
    Matrix all(y.rows()+x.rows(),d),vv=Matrix::Zero(y.rows()+x.rows(),vals.cols());all<<y,x;vv.bottomRows(x.rows())=vals;
    Permutohedral noblur(all,false);double gain=1/(1+std::ldexp(1.,-d));
    close(noblur.filter(vv,static_cast<int>(y.rows())).topRows(y.rows()),gain*cached,1e-11,"two no-blur amplitude conventions");
    auto cpd=posterior_statistics(x,y,1,0,false);auto inverse=posterior_statistics(x,y,1,0,true);
    check(std::abs(cpd.mass-x.rows())<1e-11,"CPD mass orientation");
    close(inverse.rho,Vector::Ones(y.rows()),1e-12,"FilterReg row probability");
    check(basis(y,10).cols()==(d==2?66:286),"degree ten supported");
    auto identity=paired(y);AnalyticOptions ao;auto fit=fit_analytic(y,identity,3,ao);
    close(fit.next,y,1e-11,"unregularized identity fitting");
    auto powers=exponents(d,3);Matrix coefficients=Matrix::Zero(powers.size(),d);
    for(int a=0;a<d;++a)coefficients(1+a,a)=1;
    coefficients(0,0)=.17;coefficients(d+1,0)=.3;
    Matrix target=basis(y,3)*coefficients;auto nonlinear=fit_analytic(y,paired(target),3,ao);
    close(nonlinear.next,target,1e-11,"nonlinear weighted analytic fitting");
    check(variance_from_statistics(target,paired(target),1e-12)==1e-12,"zero residual floor");
    const int parameters=d==2?3:6;auto jac=twist_jacobian(point);
    for(int k=0;k<parameters;++k){Vector delta=Vector::Zero(parameters);delta[k]=1e-7;
        auto p=twist_pose(delta,d),q=twist_pose(-delta,d);
        Vector derivative=((p.rotation*point+p.translation)-(q.rotation*point+q.translation))/(2e-7);
        close(derivative,jac.col(k),1e-8,"twist finite difference");}
    Vector delta=Vector::Constant(parameters,.015);auto pose=twist_pose(delta,d);
    auto z=apply_pose(y,pose.rotation,pose.translation);auto stats=paired(z);
    FilterOptions fo;fo.inner_iterations=5;auto rigid=fit_rigid(y,stats,fo);
    close(rigid.next,z,1e-10,"twist known rigid mapping");
    fo.solver="kabsch";close(fit_rigid(y,stats,fo).next,z,1e-10,"weighted Kabsch mapping");
    stats.normals=cloud(y.rows()+7,d).bottomRows(y.rows());for(int i=0;i<y.rows();++i)stats.normals.row(i).normalize();
    fo.solver="twist";fo.objective="point_to_plane";
    close(fit_rigid(y,stats,fo).next,z,1e-9,"point-to-plane/line twist");
    Options o;o.method=Method::Rigid;o.backend=Backend::Direct;o.rigid.sigma2=.015;
    const Matrix eye=Matrix::Identity(d,d);const Vector zero=Vector::Zero(d);
    auto rigid_result=registration(z,y,o,eye,zero);
    o.method=Method::Nonrigid;o.analytic.max_degree=2;o.analytic.max_iterations=12;
    auto hybrid=registration(z,y,o,eye,zero);
    close(hybrid.rotation,rigid_result.rotation,1e-13,"frozen hybrid rotation");
    close(hybrid.translation,rigid_result.translation,1e-13,"frozen hybrid translation");
    close(hybrid.apply(y),hybrid.transformed,1e-10,"saved composition exactly matches returned state");
    check(hybrid.steps.size()==static_cast<std::size_t>(hybrid.analytic_stage.best_iteration),"best state and map length match");
    o.method=Method::Rigid;o.backend=Backend::PermutohedralNoBlur;o.rigid.sigma2=.7;o.rigid.update_sigma2=false;o.rigid.max_iterations=4;
    auto reuse=registration(z,y,o,eye,zero);check(reuse.rigid_stage.index_builds==1,"fixed variance index reuse");
    throws([&]{Permutohedral l(y,true);l.filter(Matrix::Ones(2,1));},"invalid lattice values");
    throws([&]{auto bad=point;bad[0]=1e30;enclosing_simplex(bad.transpose(),true);},"lattice overflow guard");
    throws([&]{posterior_statistics(x,y,1,.1,false,Backend::Permutohedral);},"no accelerated posterior substitution");
    throws([&]{posterior_statistics(x,y,-1,.1,false);},"invalid variance");
    throws([&]{validate_normals(Matrix::Zero(x.rows(),d),x);},"zero normal rejected");
    auto bad=y;bad(0,0)=std::numeric_limits<double>::quiet_NaN();
    throws([&]{registration(x,bad,o,eye,zero);},"NaN rejected");
}
}
int main(){try{
    for(int d:{2,3})dimension_tests(d);
    auto s=degree_schedule(55,1,10);check(s.size()==55,"schedule size");
    for(int q=1;q<=10;++q)check(std::count(s.begin(),s.end(),q)==11-q,"triangular stages");
    check(degree_schedule(3,1,10)==std::vector<int>({1,2,3}),"short budget remainder");
    throws([]{backend_from_string("grid");},"old substitute backend rejected");
    throws([]{FilterOptions o;o.objective="point_to_plane";o.solver="kabsch";o.validate();},"invalid solver combination");
    throws([]{Options o;o.method=Method::Analytic;o.analytic.initialization="filterreg";o.validate();},"invalid variance inheritance");
    std::cout<<checks<<" native assertions passed (active also in Release)\n";return 0;
}catch(const std::exception& e){std::cerr<<"failed after "<<checks<<" assertions: "<<e.what()<<'\n';return 1;}}
