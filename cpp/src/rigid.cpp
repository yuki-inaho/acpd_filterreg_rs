#include "acpd/rigid.hpp"
#include <Eigen/SVD>
#include <Eigen/LU>
#include <cmath>
namespace acpd {
    Matrix twist_jacobian(const Vector& p) {
        if((p.size()!=2 && p.size()!=3) || !p.allFinite()) throw std::invalid_argument("twist point needs 2/3 finite coordinates");
        if(p.size()==2) {
            Matrix j(2,3);
            j<<-p[1],1,0,p[0],0,1;
            return j;
        }
        Matrix j=Matrix::Zero(3,6);
        j(0,1)=p[2];
        j(0,2)=-p[1];
        j(1,0)=-p[2];
        j(1,2)=p[0];
        j(2,0)=p[1];
        j(2,1)=-p[0];
        j.rightCols(3).setIdentity();
        return j;
    }
    Pose twist_pose(const Vector& delta,int d) {
        if((d!=2 && d!=3) || delta.size()!=(d==2?3:6) || !delta.allFinite()) throw std::invalid_argument("invalid twist size/values");
        Matrix r=Matrix::Identity(d,d);
        if(d==2) {
            const double c=std::cos(delta[0]),s=std::sin(delta[0]);
            r<<c,-s,s,c;
        }
        else {
            const Vector w=delta.head(3);
            const double theta=w.norm();
            Matrix a(3,3);
            a<<0,-w[2],w[1],w[2],0,-w[0],-w[1],w[0],0;
            const double theta2=theta*theta;
            const double sinc=theta<1e-4?1-theta2/6+theta2*theta2/120:std::sin(theta)/theta;
            const double cosc=theta<1e-4?0.5-theta2/24+theta2*theta2/720:(1-std::cos(theta))/theta2;
            r+=sinc*a+cosc*a*a;
        }
        // Matches original FilterReg / probreg: rotation-vector retraction plus a
        // translation increment, left-composed. This is not the full SE(d) exp V*v.
        return {
            r,delta.tail(d)
        };
    }
    RigidFit fit_rigid(const Matrix& y,const Statistics& s,const FilterOptions& o) {
        validate_cloud(y,"rigid points");
        o.validate();
        const int d=static_cast<int>(y.cols());
        if(s.rho.size()!=y.rows() || s.px.rows()!=y.rows() || s.px.cols()!=d || !s.rho.allFinite() || !s.px.allFinite() || !std::isfinite(s.mass) || s.mass<=1e-14 || (s.rho.array()<0).any())
        throw NumericalError("invalid or insufficient FilterReg posterior mass");
        const bool plane=o.objective=="point_to_plane";
        if(plane && (s.normals.rows()!=y.rows() || s.normals.cols()!=d || !s.normals.allFinite()))
        throw std::invalid_argument("point-to-plane requires filtered target normals");
        std::vector<int> active;
        for(int i=0;i<y.rows();++i) if(s.rho[i]>1e-12 && (!plane || s.normals.row(i).norm()>1e-12)) active.push_back(i);
        const int n=static_cast<int>(active.size());
        if(n<d) throw NumericalError("insufficient effective FilterReg correspondences");
        Matrix rotation=Matrix::Identity(d,d),next=y;
        Vector translation=Vector::Zero(d);
        int rank=0;
        if(o.solver=="kabsch") {
            // Exact minimizer of Eq.(7) for global rigid point-to-point alignment.
            // Unlike probreg's kabsch.cc, use the SAME weights for centroids and H.
            double mass=0;
            Vector cy=Vector::Zero(d),cz=Vector::Zero(d);
            for(int i:active) {
                mass+=s.rho[i];
                cy+=s.rho[i]*y.row(i).transpose();
                cz+=s.px.row(i).transpose();
            }
            cy/=mass;
            cz/=mass;
            Matrix h=Matrix::Zero(d,d);
            for(int i:active) h+=(y.row(i).transpose()-cy)*(s.px.row(i)-s.rho[i]*cz.transpose());
            Eigen::JacobiSVD<Matrix> svd(h,Eigen::ComputeFullU|Eigen::ComputeFullV);
            svd.setThreshold(1e-12);
            rank=static_cast<int>(svd.rank());
            if(rank<d-1) throw NumericalError("rigid pose is not identifiable");
            Matrix sign=Matrix::Identity(d,d);
            if((svd.matrixV()*svd.matrixU().transpose()).determinant()<0) sign(d-1,d-1)=-1;
            rotation=svd.matrixV()*sign*svd.matrixU().transpose();
            translation=cz-rotation*cy;
            next=apply_pose(y,rotation,translation);
        } else {
            const int parameters=d==2?3:6;
            for(int iteration=0;iteration<o.inner_iterations;++iteration) {
                // Paper Eq.(18): accumulate only a 3x3 / 6x6 twist system. Do not
                // allocate a point-count-sized Jacobian or pose-parameter matrix.
                Matrix normal=Matrix::Zero(parameters,parameters);
                Vector gradient=Vector::Zero(parameters);
                for(int i:active) {
                    const Matrix jp=twist_jacobian(next.row(i).transpose());
                    const Vector residual=next.row(i).transpose()-s.px.row(i).transpose()/s.rho[i];
                    if(plane) {
                        const Eigen::RowVectorXd projected=s.normals.row(i)*jp;
                        const double value=s.normals.row(i).dot(residual.transpose());
                        normal.noalias()+=s.rho[i]*projected.transpose()*projected;
                        gradient.noalias()+=s.rho[i]*projected.transpose()*value;
                    } else {
                        normal.noalias()+=s.rho[i]*jp.transpose()*jp;
                        gradient.noalias()+=s.rho[i]*jp.transpose()*residual;
                    }
                }
                require_finite(normal,"twist normal matrix");
                require_finite(gradient,"twist gradient");
                Eigen::JacobiSVD<Matrix> solver(normal,Eigen::ComputeFullU|Eigen::ComputeFullV);
                solver.setThreshold(1e-12);
                rank=static_cast<int>(solver.rank());
                if(rank<parameters) throw NumericalError("twist normal system is rank deficient");
                const Vector delta=solver.solve(-gradient);
                const Pose step=twist_pose(delta,d);
                next=apply_pose(next,step.rotation,step.translation);
                rotation=step.rotation*rotation;
                translation=step.rotation*translation+step.translation;
                if(delta.norm()<o.tolerance) break;
            }
        }
        double error=0,mass=0;
        for(int i:active) {
            const Eigen::RowVectorXd r=next.row(i)-s.px.row(i)/s.rho[i];
            error+=s.rho[i]*(plane?std::pow(s.normals.row(i).dot(r),2):r.squaredNorm());
            mass+=s.rho[i];
        }
        return {
            rotation,next,translation,rank,n,std::sqrt(error/mass)
        };
    }
}
// namespace acpd
