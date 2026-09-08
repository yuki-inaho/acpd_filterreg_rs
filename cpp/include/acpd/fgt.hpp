#pragma once
#include "types.hpp"
namespace acpd {
    /// Improved Fast Gauss Transform: cluster the sources, expand the Gaussian in a
    /// truncated Taylor series about each cluster centre, evaluate per query.
    ///
    ///   exp(-|q-s|^2/h^2) = exp(-|q-c|^2/h^2) exp(-|s-c|^2/h^2)
    ///                       sum_alpha (2^|alpha|/alpha!) ((q-c)/h)^alpha ((s-c)/h)^alpha
    ///
    /// with h^2 = 2 sigma^2, so the kernel matches the exact exp(-|q-s|^2/(2 sigma^2))
    /// used everywhere else. This is an approximation with an explicit, reported error
    /// bound; it is never selected implicitly and never substituted for a failed exact
    /// computation. Not part of either source paper's own implementation.
    struct FgtCost {
        int clusters = 0, terms = 0;
        long long cluster_query_pairs = 0;
        double covering_radius_over_h = 0;
        bool radius_target_met = false;
        // false means max_clusters was reached before the requested covering radius,
        // so the truncation error is larger than the setting asks for.
    };
    Matrix fgt_transform(const Matrix& sources, const Matrix& queries, const Matrix& values,
    double sigma2, const FgtOptions& options, FgtCost* cost = nullptr);
}
// namespace acpd
