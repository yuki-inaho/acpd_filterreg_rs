#pragma once
#include "types.hpp"
#include <string>
namespace acpd {
    /// GPU Gaussian-sum operator for the E-step.
    ///
    /// Both directions reduce to one operator,
    ///   out(q, ch) = sum_s exp(-|q - s|^2 / (2 sigma^2)) * values(s, ch),
    /// evaluated once for the FilterReg direction and twice for the CPD direction
    /// (forward denominators, then the transpose moments). The correspondence
    /// matrix is never formed; only the operator's inputs and outputs move.
    ///
    /// This is exact pair evaluation, not an approximation. It is selected
    /// explicitly and never substituted for a failed exact computation: a build
    /// without CUDA, or a machine without a device, raises instead of computing
    /// on the CPU.
    bool cuda_available();
    std::string cuda_device_name();
    Matrix cuda_gaussian_transform(const Matrix& sources, const Matrix& queries,
    const Matrix& values, double sigma2, const CudaOptions& options);
}
// namespace acpd
