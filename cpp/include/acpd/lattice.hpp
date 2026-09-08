#pragma once
#include "types.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>
namespace acpd {
    using LatticeKey = std::vector<std::int64_t>;
    struct LatticeKeyHash {
        std::size_t operator()(const LatticeKey& key) const noexcept;
    };
    struct Simplex {
        std::vector<LatticeKey> keys;
        std::vector<double> weights;
    };
    /// Scalar, double-precision port of the simplex algorithm in probreg's BSD
    /// permutohedral implementation. Integer keys are widened to avoid int16 wrap.
    Simplex enclosing_simplex(const Eigen::Ref<const Eigen::RowVectorXd>& feature, bool with_blur);
    /// Splat -> (optional all-axis Blur) -> Slice, with probreg's output gain.
    /// Features are already whitened (positions / sigma). No spatial radius search.
    class Permutohedral {
        public:
        Permutohedral(const Matrix& features, bool with_blur = true);
        Matrix filter(const Matrix& values, int start = 0, bool reverse = false) const;
        std::size_t lattice_size() const {
            return keys_.size();
        }
        bool with_blur() const {
            return with_blur_;
        }
        private:
        int n_, d_;
        bool with_blur_;
        std::vector<LatticeKey> keys_;
        std::vector<std::size_t> offsets_;
        std::vector<double> barycentric_;
        std::vector<std::pair<int,int>> neighbors_;
    };
    /// Original FilterReg specialization: observations splat once, queries ONLY slice.
    /// No blur, no probreg alpha multiplier; immutable after construction, reusable
    /// across iterations at fixed observation features, values, and variance.
    class FixedNoBlurLattice {
        public:
        FixedNoBlurLattice(const Matrix& features, const Matrix& values);
        Matrix slice(const Matrix& query_features) const;
        std::size_t lattice_size() const {
            return keys_.size();
        }
        private:
        int d_;
        std::vector<LatticeKey> keys_;
        std::unordered_map<LatticeKey,std::size_t,LatticeKeyHash> index_;
        Matrix splatted_;
    };
    struct FilteredValues {
        Matrix values;
        int vertices = 0;
        std::string mode = "direct";
    };
    FilteredValues lattice_transform(const Matrix& sources, const Matrix& queries,
    const Matrix& values, double sigma2, Backend backend);
}
// namespace acpd
