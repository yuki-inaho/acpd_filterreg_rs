#include "acpd/fgt.hpp"
#include "acpd/lattice.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>
namespace acpd {
    namespace {
        /// Multi-indices alpha with |alpha| <= order, and their 2^|alpha|/alpha! factor.
        struct Terms {
            std::vector<std::vector<int>> exponents;
            std::vector<double> factors;
        };
        Terms build_terms(int d,int order) {
            Terms out;
            std::vector<int> alpha(static_cast<std::size_t>(d),0);
            // Odometer over all alpha with total degree <= order.
            while(true) {
                int total=0;
                for(int a=0;a<d;++a) total+=alpha[static_cast<std::size_t>(a)];
                if(total<=order) {
                    double factorial=1;
                    for(int a=0;a<d;++a) for(int j=2;j<=alpha[static_cast<std::size_t>(a)];++j) factorial*=j;
                    out.exponents.push_back(alpha);
                    out.factors.push_back(std::ldexp(1.0,total)/factorial);
                }
                int axis=d-1;
                while(axis>=0&&alpha[static_cast<std::size_t>(axis)]==order) {
                    alpha[static_cast<std::size_t>(axis)]=0;
                    --axis;
                }
                if(axis<0) break;
                ++alpha[static_cast<std::size_t>(axis)];
            }
            return out;
        }
        /// Uniform-grid clustering in O(n): sources are bucketed into axis-aligned cells
        /// whose half-diagonal is the requested covering radius, so every source is within
        /// `target` of its cell centre by construction. Farthest-point clustering would give
        /// tighter centres but costs O(k n), which degenerates to O(n^2) once sigma is small
        /// enough that nearly every source needs its own centre.
        struct Clustering {
            Matrix centre;
            std::vector<int> owner;
            double radius = 0;
            bool radius_met = true;
        };
        Clustering grid_clusters(const Matrix& points,int cap,double target) {
            const int n=static_cast<int>(points.rows()),d=static_cast<int>(points.cols());
            const double cell=2*target/std::sqrt(double(d));
            Clustering out;
            out.owner.assign(static_cast<std::size_t>(n),0);
            std::unordered_map<LatticeKey,int,LatticeKeyHash> index;
            index.reserve(static_cast<std::size_t>(n));
            std::vector<LatticeKey> cells;
            LatticeKey key(d);
            for(int i=0;i<n;++i) {
                for(int a=0;a<d;++a) {
                    const double scaled=std::floor(points(i,a)/cell);
                    if(!std::isfinite(scaled)||std::abs(scaled)>0x1p46)
                    throw NumericalError("fgt grid coordinate overflow: rescale points or raise cluster_radius");
                    key[a]=static_cast<std::int64_t>(scaled);
                }
                auto inserted=index.emplace(key,static_cast<int>(cells.size()));
                if(inserted.second) {
                    if(static_cast<int>(cells.size())>=cap) {
                        // Cap reached: fall back to one coarse cluster set rather than
                        // silently mixing resolutions. Reported through radius_met.
                        out.radius_met=false;
                        index.erase(key);
                        out.owner[static_cast<std::size_t>(i)]=0;
                        continue;
                    }
                    cells.push_back(key);
                }
                out.owner[static_cast<std::size_t>(i)]=inserted.first->second;
            }
            const int clusters=static_cast<int>(cells.size());
            out.centre.resize(clusters,d);
            for(int c=0;c<clusters;++c) for(int a=0;a<d;++a)
            out.centre(c,a)=(static_cast<double>(cells[static_cast<std::size_t>(c)][a])+0.5)*cell;
            double worst=0;
            for(int i=0;i<n;++i)
            worst=std::max(worst,(points.row(i)-out.centre.row(out.owner[static_cast<std::size_t>(i)])).squaredNorm());
            out.radius=std::sqrt(worst);
            return out;
        }
    }
    Matrix fgt_transform(const Matrix& s,const Matrix& q,const Matrix& v,double sigma2,
    const FgtOptions& options,FgtCost* cost) {
        options.validate();
        if(s.rows()==0||q.rows()==0||s.cols()!=q.cols()||s.rows()!=v.rows()||v.cols()<1
        ||!s.allFinite()||!q.allFinite()||!v.allFinite()||!std::isfinite(sigma2)||sigma2<=0)
        throw std::invalid_argument("invalid fast Gauss transform arguments");
        const int d=static_cast<int>(s.cols()),n=static_cast<int>(s.rows()),m=static_cast<int>(q.rows());
        const int columns=static_cast<int>(v.cols());
        const double h=std::sqrt(2*sigma2);
        const Terms terms=build_terms(d,options.order);
        const int width=static_cast<int>(terms.exponents.size());
        const Clustering clustering=grid_clusters(s,std::min(options.max_clusters,n),options.cluster_radius*h);
        const auto& owner=clustering.owner;
        const Matrix& centre=clustering.centre;
        const int clusters=static_cast<int>(centre.rows());
        // Coefficients C[(cluster*width + term), column].
        Matrix coefficients=Matrix::Zero(clusters*width,columns);
        std::vector<double> power(static_cast<std::size_t>(d)*(options.order+1),0.0);
        const auto monomials=[&](const Eigen::RowVectorXd& u,std::vector<double>& into) {
            for(int a=0;a<d;++a) {
                power[static_cast<std::size_t>(a)*(options.order+1)]=1;
                for(int k=1;k<=options.order;++k)
                power[static_cast<std::size_t>(a)*(options.order+1)+k]
                =power[static_cast<std::size_t>(a)*(options.order+1)+k-1]*u[a];
            }
            for(int t=0;t<width;++t) {
                double value=terms.factors[static_cast<std::size_t>(t)];
                for(int a=0;a<d;++a)
                value*=power[static_cast<std::size_t>(a)*(options.order+1)
                +terms.exponents[static_cast<std::size_t>(t)][static_cast<std::size_t>(a)]];
                into[static_cast<std::size_t>(t)]=value;
            }
        };
        std::vector<double> basis(static_cast<std::size_t>(width));
        for(int i=0;i<n;++i) {
            const int c=owner[static_cast<std::size_t>(i)];
            const Eigen::RowVectorXd u=(s.row(i)-centre.row(c))/h;
            const double weight=std::exp(-u.squaredNorm());
            monomials(u,basis);
            for(int t=0;t<width;++t)
            coefficients.row(c*width+t)+=(weight*basis[static_cast<std::size_t>(t)])*v.row(i);
        }
        // The Taylor factor is symmetric in the source and query monomials, so it is
        // folded into the source coefficients only and the query side uses plain powers.
        const Terms& saved=terms;
        Matrix out=Matrix::Zero(m,columns);
        const double cutoff=options.cutoff_radius*options.cutoff_radius;
        long long pairs=0;
        std::vector<double> query_basis(static_cast<std::size_t>(width));
        for(int i=0;i<m;++i) {
            for(int c=0;c<clusters;++c) {
                const Eigen::RowVectorXd z=(q.row(i)-centre.row(c))/h;
                const double distance=z.squaredNorm();
                // Prune before building the power table: a pruned cluster costs only d flops.
                if(distance>cutoff) continue;
                ++pairs;
                const double gaussian=std::exp(-distance);
                for(int a=0;a<d;++a) {
                    power[static_cast<std::size_t>(a)*(options.order+1)]=1;
                    for(int k=1;k<=options.order;++k)
                    power[static_cast<std::size_t>(a)*(options.order+1)+k]
                    =power[static_cast<std::size_t>(a)*(options.order+1)+k-1]*z[a];
                }
                for(int t=0;t<width;++t) {
                    double value=1;
                    for(int a=0;a<d;++a)
                    value*=power[static_cast<std::size_t>(a)*(options.order+1)
                    +saved.exponents[static_cast<std::size_t>(t)][static_cast<std::size_t>(a)]];
                    query_basis[static_cast<std::size_t>(t)]=gaussian*value;
                }
                for(int t=0;t<width;++t)
                out.row(i)+=query_basis[static_cast<std::size_t>(t)]*coefficients.row(c*width+t);
            }
        }
        require_finite(out,"fast Gauss transform");
        if(cost) *cost={
            clusters,width,pairs,clustering.radius/h,clustering.radius_met
        };
        return out;
    }
}
// namespace acpd
