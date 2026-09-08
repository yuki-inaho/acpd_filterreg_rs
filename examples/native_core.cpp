// Reusable native C++ example: link acpd::acpd_core; no Python dependency.
#include <acpd/types.hpp>
#include <cmath>
#include <iostream>
int main(int argc, char** argv) {
    try {
        const int d = argc > 1 ? std::stoi(argv[1]) : 3;
        if (d != 2 && d != 3) throw std::invalid_argument("dimension must be 2 or 3");
        acpd::Matrix moving(80, d);
        for (int i = 0; i < moving.rows(); ++i)
            for (int a = 0; a < d; ++a)
                moving(i, a) = std::sin(0.7*i+0.9*a)+0.3*std::cos(i*(a+1)*0.17);
        acpd::Matrix fixed = moving;
        for (int i = 0; i < fixed.rows(); ++i)
            fixed(i, 0) += 0.04 + 0.02*moving(i, 1)*moving(i, 1);
        acpd::Options options;
        options.method = acpd::Method::Nonrigid;
        options.backend = acpd::Backend::Permutohedral;
        options.rigid.sigma2 = 0.08;
        auto result = acpd::registration(fixed, moving, options,
            acpd::Matrix::Identity(d, d), acpd::Vector::Zero(d));
        std::cout << "returned analytic steps: " << result.steps.size() << '\n'
                  << "analytic stop: " << result.analytic_stage.stop_reason << '\n'
                  << "map reproduction error: " << (result.apply(moving)-result.transformed).norm() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
