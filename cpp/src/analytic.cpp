#include "acpd/analytic.hpp"
#include <Eigen/SVD>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
namespace acpd {
    std::vector<std::array<int,3>> exponents(int d,int degree) {
        if((d!=2 && d!=3) || degree<0 || degree>10) throw std::invalid_argument("basis requires dimension 2/3 and degree 0..10");
        std::vector<std::array<int,3>> out;
        for(int r=0;r<=degree;++r) for(int a=r;a>=0;--a) {
            if(d==2) out.push_back({
                a,r-a,0
            });
            else for(int b=r-a;b>=0;--b) out.push_back({
                a,b,r-a-b
            });
        }
        return out;
    }
    Matrix basis(const Matrix& points,int degree) {
        validate_cloud(points,"basis points");
        const auto powers=exponents(static_cast<int>(points.cols()),degree);
        Matrix phi(points.rows(),static_cast<int>(powers.size()));
        for(int i=0;i<points.rows();++i) for(int k=0;k<phi.cols();++k) {
            double term=1;
            for(int a=0;a<points.cols();++a) for(int j=1;j<=powers[k][a];++j) term*=points(i,a)/j;
            phi(i,k)=term;
        }
        require_finite(phi,"Taylor basis");
        return phi;
    }
    std::vector<int> degree_schedule(int iterations,int low,int high) {
        if(iterations<1 || low<1 || high<low || high>10) throw std::invalid_argument("invalid degree schedule");
        const int count=high-low+1,unit=count*(count+1)/2;
        std::vector<int> lengths(count);
        for(int k=0;k<count;++k) lengths[k]=(count-k)*(iterations/unit);
        // Exact port of Algo.h/DegreeScheduleDecreasingStages's remainder rule.
        int remaining=iterations%unit;
        for(int prefix=count;prefix>=1 && remaining>0;--prefix) {
            const int take=std::min(prefix,remaining);
            for(int k=0;k<take;++k) ++lengths[k];
            remaining-=take;
        }
        std::vector<int> out;
        out.reserve(iterations);
        for(int k=0;k<count;++k) out.insert(out.end(),lengths[k],low+k);
        return out;
    }
    Fit fit_analytic(const Matrix& y,const Statistics& stats,int degree,const AnalyticOptions& o) {
        validate_cloud(y,"analytic points");
        o.validate();
        if(stats.rho.size()!=y.rows() || stats.px.rows()!=y.rows() || stats.px.cols()!=y.cols() || !stats.rho.allFinite() || !stats.px.allFinite())
        throw std::invalid_argument("invalid analytic statistics");
        if(degree<o.min_degree || degree>o.max_degree) throw std::invalid_argument("requested degree outside options");
        std::vector<int> active;
        for(int i=0;i<y.rows();++i) {
            if(stats.rho[i]<0) throw std::invalid_argument("negative analytic fitting weight");
            if(stats.rho[i]>o.min_mass) active.push_back(i);
        }
        const int n=static_cast<int>(active.size()),d=static_cast<int>(y.cols());
        while(degree>o.min_degree && static_cast<int>(exponents(d,degree).size())>n) --degree;
        const Matrix phi=basis(y,degree);
        const int k=static_cast<int>(phi.cols());
        if(n<k) throw NumericalError("insufficient active rows for the minimum analytic degree");
        Matrix design(n,k),targets(n,d);
        double mass=0;
        for(int row=0;row<n;++row) {
            const int i=active[row];
            const double sw=std::sqrt(stats.rho[i]);
            mass+=stats.rho[i];
            design.row(row)=sw*phi.row(i);
            targets.row(row)=(sw/stats.rho[i])*stats.px.row(i);
        }
        // Paper Eq.(20)/(22): no hidden ridge penalty, coefficient projection,
        // displacement cap, or damping. SVD avoids explicitly forming Phi^T W Phi.
        require_finite(design,"weighted Taylor design");
        require_finite(targets,"weighted targets");
        // Rank-revealing complete orthogonal decomposition. Like the SVD it returns the
        // MINIMUM-NORM least-squares solution under rank deficiency and reports a rank
        // against the same relative threshold, so Eq.(20)/(22) is still solved without
        // regularization, damping or projection - only the factorization differs.
        // Measured on the real weighted Taylor designs: identical rank in every case and
        // fitted values agreeing to 4.6e-12, at 4.6x (degree 6) to 9.3x (degree 10) the
        // speed of BDCSVD. The coefficients themselves can differ more because the
        // monomial basis reaches condition 2e10 at degree 10; the algorithm consumes the
        // fitted values, and the stored map is evaluated on the same basis.
        Eigen::CompleteOrthogonalDecomposition<Matrix> solver(design);
        solver.setThreshold(o.rank_tolerance);
        Matrix absolute=solver.solve(targets);
        Matrix next=phi*absolute;
        require_finite(next,"analytic M-step");
        Matrix correction=absolute;
        for(int a=0;a<d;++a) correction(1+a,a)-=1;
        // exact representation A(p)=p+Phi(p)C
        double sse=0;
        for(int i:active) sse+=stats.rho[i]*(stats.px.row(i)/stats.rho[i]-next.row(i)).squaredNorm();
        return {
            AnalyticStep {
                degree,correction
            },next,n,static_cast<int>(solver.rank()),std::sqrt(sse/mass)
        };
    }
}
// namespace acpd
