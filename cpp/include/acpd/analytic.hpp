#pragma once
#include "gaussian.hpp"
#include <array>
namespace acpd {
    std::vector<std::array<int,3>> exponents(int dimension,int degree);
    Matrix basis(const Matrix& points,int degree);
    std::vector<int> degree_schedule(int iterations,int min_degree,int max_degree);
    struct Fit {
        AnalyticStep step;
        Matrix next;
        int active,rank;
        double fit_rms;
    };
    Fit fit_analytic(const Matrix& current,const Statistics& statistics,int requested_degree,
    const AnalyticOptions& options);
}
// namespace acpd
