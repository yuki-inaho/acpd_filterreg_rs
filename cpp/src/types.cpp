#include "acpd/types.hpp"
#include "acpd/analytic.hpp"
#include <Eigen/LU>
#include <cmath>
namespace acpd {
    namespace {
        void positive(double v,const char* label) {
            if(!std::isfinite(v)||v<=0) throw std::invalid_argument(std::string(label)+" must be finite and positive");
        }
        void common(int n,double tolerance,double w,double sigma2,double floor) {
            if(n<1||n>100000) throw std::invalid_argument("max_iterations must be in [1,100000]");
            positive(tolerance,"tolerance");
            positive(floor,"min_sigma2");
            if(!std::isfinite(w)||w<0||w>=1) throw std::invalid_argument("w must be in [0,1)");
            if(sigma2!=-1) positive(sigma2,"sigma2");
        }
    }
    Backend backend_from_string(const std::string& s) {
        if(s=="direct") return Backend::Direct;
        if(s=="permutohedral") return Backend::Permutohedral;
        if(s=="permutohedral_noblur") return Backend::PermutohedralNoBlur;
        if(s=="probreg") return Backend::Probreg;
        throw std::invalid_argument("backend must be direct, permutohedral, permutohedral_noblur, or probreg; no fallback");
    }
    std::string name(Backend b) {
        switch(b) {
            case Backend::Direct:return "direct";
            case Backend::Permutohedral:return "permutohedral";
            case Backend::PermutohedralNoBlur:return "permutohedral_noblur";
            case Backend::Probreg:return "probreg";
        }
        throw std::invalid_argument("invalid backend enum");
    }
    Method method_from_string(const std::string& s) {
        if(s=="rigid") return Method::Rigid;
        if(s=="analytic") return Method::Analytic;
        if(s=="nonrigid") return Method::Nonrigid;
        throw std::invalid_argument("method must be rigid, analytic, or nonrigid");
    }
    std::string name(Method m) {
        switch(m) {
            case Method::Rigid: return "rigid";
            case Method::Analytic: return "analytic";
            case Method::Nonrigid: return "nonrigid";
        }
        throw std::invalid_argument("invalid method enum");
    }
    void FilterOptions::validate() const {
        common(max_iterations,tolerance,w,sigma2,min_sigma2);
        if(solver!="twist"&&solver!="kabsch") throw std::invalid_argument("solver must be twist or kabsch");
        if(objective!="point_to_point"&&objective!="point_to_plane") throw std::invalid_argument("invalid rigid objective");
        if(objective=="point_to_plane"&&solver!="twist") throw std::invalid_argument("point-to-plane requires twist solver");
        if(inner_iterations<1||inner_iterations>100) throw std::invalid_argument("inner_iterations must be in [1,100]");
    }
    void AnalyticOptions::validate() const {
        common(max_iterations,tolerance,w,sigma2,min_sigma2);
        if(min_degree<1||max_degree>10||min_degree>max_degree) throw std::invalid_argument("analytic degrees require 1<=min<=max<=10");
        positive(rank_tolerance,"rank_tolerance");
        if(rank_tolerance>=1) throw std::invalid_argument("rank_tolerance must be <1");
        positive(min_mass,"min_mass");
        if(initialization!="auto"&&initialization!="cpd"&&initialization!="filterreg") throw std::invalid_argument("invalid analytic initialization");
        if(stable_patience<1||no_improve_patience<1||min_iterations<1) throw std::invalid_argument("stopping counts must be positive");
        if(!std::isfinite(improvement_relative)||improvement_relative<0||!std::isfinite(rebound_relative)||rebound_relative<0)
        throw std::invalid_argument("relative stopping thresholds must be finite and nonnegative");
        if(!std::isfinite(divergence_radius)||divergence_radius<=1)
        throw std::invalid_argument("divergence_radius must be finite and greater than one");
    }
    void Options::validate() const {
        name(method);
        name(backend);
        rigid.validate();
        analytic.validate();
        if(method==Method::Analytic&&analytic.initialization=="filterreg"&&analytic.sigma2<0)
        throw std::invalid_argument("standalone analytic mode has no FilterReg variance to inherit");
    }
    void require_finite(const Matrix& v,const char* c) {
        if(!v.allFinite()) throw NumericalError(std::string(c)+": non-finite result");
    }
    void validate_cloud(const Matrix& p,const std::string& label) {
        if(p.rows()==0||(p.cols()!=2&&p.cols()!=3)||!p.allFinite()) throw std::invalid_argument(label+" needs finite nonempty shape (n,2) or (n,3)");
    }
    void validate_pair(const Matrix& x,const Matrix& y,bool reg) {
        validate_cloud(x,"fixed");
        validate_cloud(y,"moving");
        if(x.cols()!=y.cols()) throw std::invalid_argument("point dimensions mismatch");
        if(reg&&(x.rows()<x.cols()+1||y.rows()<y.cols()+1)) throw std::invalid_argument("registration requires at least d+1 points per cloud");
    }
    void validate_pose(const Matrix& r,const Vector& t,int d) {
        if(r.rows()!=d||r.cols()!=d||t.size()!=d||!r.allFinite()||!t.allFinite()) throw std::invalid_argument("invalid initial pose dimensions/values");
        if((r.transpose()*r-Matrix::Identity(d,d)).norm()>1e-8||std::abs(r.determinant()-1)>1e-8) throw std::invalid_argument("initial rotation must be in SO(d)");
    }
    void validate_normals(const Matrix& normals,const Matrix& fixed) {
        if(normals.rows()!=fixed.rows()||normals.cols()!=fixed.cols()||!normals.allFinite()) throw std::invalid_argument("target normals must match fixed points");
        for(int i=0;i<normals.rows();++i) if(std::abs(normals.row(i).norm()-1)>1e-6) throw std::invalid_argument("each target normal must have unit norm");
    }
    Matrix apply_pose(const Matrix& p,const Matrix& r,const Vector& t) {
        validate_cloud(p,"pose points");
        validate_pose(r,t,static_cast<int>(p.cols()));
        Matrix out=p*r.transpose();
        out.rowwise()+=t.transpose();
        require_finite(out,"pose");
        return out;
    }
    Matrix Result::apply(const Matrix& p) const {
        validate_cloud(p,"points");
        if(p.cols()!=rotation.cols()) throw std::invalid_argument("transform dimension mismatch");
        Matrix u=apply_pose(p,rotation,translation);
        u.rowwise()-=center.transpose();
        u/=normalization_scale;
        for(const auto& step:steps) {
            u+=basis(u,step.degree)*step.coefficients;
            require_finite(u,"analytic composition");
        }
        u*=normalization_scale;
        u.rowwise()+=center.transpose();
        require_finite(u,"denormalization");
        return u;
    }
}
// namespace acpd
