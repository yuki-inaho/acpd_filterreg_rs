#pragma once
#include "gaussian.hpp"
namespace acpd {
    struct Pose {
        Matrix rotation;
        Vector translation;
    };
    struct RigidFit {
        Matrix rotation,next;
        Vector translation;
        int rank,active;
        double fit_rms;
    };
    /// Left perturbation, parameters [theta,tx,ty] in 2D and [wx,wy,wz,tx,ty,tz] in 3D.
    Matrix twist_jacobian(const Vector& point);
    Pose twist_pose(const Vector& twist,int dimension);
    RigidFit fit_rigid(const Matrix& current,const Statistics& stats,const FilterOptions& options);
}
// namespace acpd
