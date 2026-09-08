#pragma once
#include "lattice.hpp"
namespace acpd {
    struct Statistics {
        Vector rho;
        // CPD row masses or FilterReg inlier responsibilities
        Matrix px;
        // posterior-weighted target coordinates
        Vector x2;
        // posterior-weighted squared target norm per moving point
        Matrix normals;
        // FilterReg mean normals (NOT renormalized)
        double mass = 0, nll = 0;
        int vertices = 0, unsupported = 0;
        std::string lattice_mode = "direct";
    };
    Matrix moment_values(const Matrix& fixed,const Matrix& normals=Matrix());
    Statistics posterior_statistics(const Matrix& fixed,const Matrix& current,double sigma2,
    double w,bool filterreg,Backend backend=Backend::Direct,
    const Matrix& target_normals=Matrix(),
    const FixedNoBlurLattice* cache=nullptr,
    const FgtOptions& fgt=FgtOptions(),
    const CudaOptions& cuda=CudaOptions());
    Matrix gaussian_sum(const Matrix& sources,const Matrix& queries,const Matrix& values,
    double sigma2,Backend backend=Backend::Direct,const FgtOptions& fgt=FgtOptions(),
    const CudaOptions& cuda=CudaOptions());
    double variance_from_statistics(const Matrix& next,const Statistics& stats,double floor);
    double initial_variance(const Matrix& fixed,const Matrix& moving);
}
// namespace acpd
