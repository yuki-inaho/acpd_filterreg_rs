#include "acpd/gaussian.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
namespace acpd {
    namespace {
        constexpr double pi=3.141592653589793238462643383279502884;
    }
    Matrix gaussian_sum(const Matrix& s,const Matrix& q,const Matrix& v,double sigma2,Backend b) {
        return lattice_transform(s,q,v,sigma2,b).values;
    }
    Matrix moment_values(const Matrix& x,const Matrix& normals) {
        validate_cloud(x,"fixed");
        if(normals.size()) validate_normals(normals,x);
        const int d=static_cast<int>(x.cols());
        Matrix v(x.rows(),d+2+(normals.size()?d:0));
        v.col(0).setOnes();
        v.middleCols(1,d)=x;
        for(int j=0;j<x.rows();++j) v(j,d+1)=x.row(j).squaredNorm();
        if(normals.size()) v.rightCols(d)=normals;
        require_finite(v,"Gaussian moment values");
        return v;
    }
    double initial_variance(const Matrix& x,const Matrix& y) {
        validate_pair(x,y);
        const Eigen::RowVectorXd mx=x.colwise().mean(),my=y.colwise().mean();
        double sx=0,sy=0;
        for(int i=0;i<x.rows();++i) sx+=(x.row(i)-mx).squaredNorm()/x.rows();
        for(int i=0;i<y.rows();++i) sy+=(y.row(i)-my).squaredNorm()/y.rows();
        const double result=(sx+sy+(mx-my).squaredNorm())/x.cols();
        if(!std::isfinite(result)) throw NumericalError("initial variance overflow");
        return result;
    }
    Statistics posterior_statistics(const Matrix& x,const Matrix& y,double sigma2,double w,
    bool inverse,Backend backend,const Matrix& normals,const FixedNoBlurLattice* cache) {
        validate_pair(x,y);
        if(!std::isfinite(sigma2) || sigma2<=0 || !std::isfinite(w) || w<0 || w>=1)
        throw std::invalid_argument("sigma2 must be positive and w in [0,1)");
        if(normals.size()) validate_normals(normals,x);
        if(!inverse && backend!=Backend::Direct)
        throw std::invalid_argument("paper-faithful Analytic-CPD uses direct posterior computation");
        const int m=static_cast<int>(y.rows()),n=static_cast<int>(x.rows()),d=static_cast<int>(x.cols());
        const int centers=inverse?n:m,queries=inverse?m:n;
        const double normalizer=0.5*d*(std::log(2*pi)+std::log(sigma2));
        const double logc=w==0?-std::numeric_limits<double>::infinity():normalizer+std::log(w)-std::log1p(-w)+std::log(double(centers)/queries);
        const double logfactor=std::log1p(-w)-std::log(double(centers))-normalizer;
        Statistics out;
        out.rho=Vector::Zero(m);
        out.px=Matrix::Zero(m,d);
        out.x2=Vector::Zero(m);
        if(normals.size()) out.normals=Matrix::Zero(m,d);
        if(inverse && backend!=Backend::Direct) {
            FilteredValues filtered;
            if(cache && backend==Backend::PermutohedralNoBlur)
            filtered= {
                cache->slice(y/std::sqrt(sigma2)),static_cast<int>(cache->lattice_size()),"original_noblur"
            };
            else filtered=lattice_transform(x,y,moment_values(x,normals),sigma2,backend);
            out.vertices=filtered.vertices;
            out.lattice_mode=filtered.mode;
            const Matrix& moments=filtered.values;
            for(int i=0;i<m;++i) {
                const double m0=moments(i,0);
                if(m0<0) throw NumericalError("negative permutohedral zeroth moment");
                if(m0==0) {
                    ++out.unsupported;
                    out.nll-=(w>0?std::log(w)-std::log(double(m)):-std::numeric_limits<double>::infinity());
                    continue;
                }
                // Logistic form is stable even when the outlier constant overflows.
                const double logm=std::log(m0), maxlog=std::max(logm,logc);
                const double logden=maxlog+std::log(std::exp(logm-maxlog)+std::exp(logc-maxlog));
                out.rho[i]=std::exp(logm-logden);
                out.px.row(i)=out.rho[i]*(moments.row(i).segment(1,d)/m0);
                out.x2[i]=out.rho[i]*(moments(i,d+1)/m0);
                if(normals.size()) out.normals.row(i)=moments.row(i).tail(d)/m0;
                out.nll-=logden+logfactor;
            }
        } else {
            // Direct streaming E-step: same CPD posterior as the supplied Algo.h,
            // with log-sum-exp only to avoid its 0/0 / denominator-floor edge case.
            Vector weights(centers);
            for(int q=0;q<queries;++q) {
                double maxlog=logc;
                for(int c=0;c<centers;++c) {
                    const double distance=inverse?(y.row(q)-x.row(c)).squaredNorm():(x.row(q)-y.row(c)).squaredNorm();
                    if(!std::isfinite(distance)) throw NumericalError("squared distance overflow");
                    weights[c]=-0.5*(distance/sigma2);
                    maxlog=std::max(maxlog,weights[c]);
                }
                if(!std::isfinite(maxlog)) throw NumericalError("posterior has no representable support");
                double denominator=std::exp(logc-maxlog);
                for(int c=0;c<centers;++c) {
                    weights[c]=std::exp(weights[c]-maxlog);
                    denominator+=weights[c];
                }
                out.nll-=maxlog+std::log(denominator)+logfactor;
                for(int c=0;c<centers;++c) {
                    const double p=weights[c]/denominator;
                    const int i=inverse?q:c,j=inverse?c:q;
                    out.rho[i]+=p;
                    out.px.row(i)+=p*x.row(j);
                    out.x2[i]+=p*x.row(j).squaredNorm();
                    if(normals.size()) out.normals.row(i)+=p*normals.row(j);
                }
            }
            if(normals.size()) for(int i=0;i<m;++i) if(out.rho[i]>0) out.normals.row(i)/=out.rho[i];
        }
        out.mass=out.rho.sum();
        if(!std::isfinite(out.mass) || !out.px.allFinite() || !out.x2.allFinite())
        throw NumericalError("non-finite posterior moments");
        return out;
    }
    double variance_from_statistics(const Matrix& y,const Statistics& s,double floor) {
        if(y.rows()!=s.rho.size() || s.px.rows()!=y.rows() || s.px.cols()!=y.cols() || s.x2.size()!=y.rows())
        throw std::invalid_argument("variance statistic dimensions mismatch");
        if(!std::isfinite(floor) || floor<=0) throw std::invalid_argument("variance floor must be positive");
        validate_cloud(y,"variance points");
        if(!std::isfinite(s.mass) || s.mass<=1e-14 || !s.rho.allFinite() || !s.px.allFinite() || !s.x2.allFinite() || (s.rho.array()<0).any())
        throw NumericalError("invalid or insufficient posterior mass");
        double residual=0,magnitude=0;
        for(int i=0;i<y.rows();++i) {
            const double a=s.x2[i],b=2*y.row(i).dot(s.px.row(i)),c=s.rho[i]*y.row(i).squaredNorm();
            residual+=a-b+c;
            magnitude+=std::abs(a)+std::abs(b)+std::abs(c);
        }
        if(!std::isfinite(residual) || residual < -1e-10*std::max(1.0,magnitude))
        throw NumericalError("invalid second-moment residual");
        return std::max(floor,std::max(0.0,residual)/(y.cols()*s.mass));
    }
}
// namespace acpd
