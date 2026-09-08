#pragma once
#include "types.hpp"
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>
namespace acpd {
    inline constexpr int max_lattice_dimension = 16;
    /// Inline, allocation-free lattice coordinate. The scalar upstream stores signed
    /// shorts; keys stay widened to int64 here but are held by value, so the splat,
    /// neighbour and slice loops perform no heap traffic. Values and ordering are
    /// unchanged; only the storage of a key differs from a std::vector of the same
    /// integers.
    class LatticeKey {
        public:
        LatticeKey() = default;
        explicit LatticeKey(int size): size_(size) {
            if(size<0||size>max_lattice_dimension) throw std::invalid_argument("lattice key size out of range");
        }
        int size() const noexcept {
            return size_;
        }
        std::int64_t& operator[](int index) noexcept {
            return data_[static_cast<std::size_t>(index)];
        }
        std::int64_t operator[](int index) const noexcept {
            return data_[static_cast<std::size_t>(index)];
        }
        const std::int64_t* begin() const noexcept {
            return data_.data();
        }
        const std::int64_t* end() const noexcept {
            return data_.data()+size_;
        }
        bool operator==(const LatticeKey& other) const noexcept {
            if(size_!=other.size_) return false;
            for(int i=0;i<size_;++i) if(data_[static_cast<std::size_t>(i)]!=other.data_[static_cast<std::size_t>(i)]) return false;
            return true;
        }
        private:
        std::array<std::int64_t,max_lattice_dimension> data_{};
        int size_ = 0;
    };
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
    /// Same computation, reusing the caller's storage. Hot loops call this once per
    /// point with a hoisted Simplex so no allocation happens after the first point.
    void enclosing_simplex(const Eigen::Ref<const Eigen::RowVectorXd>& feature, bool with_blur, Simplex& out);
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
        // Row-major: every splat and slice touches one whole vertex row.
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> splatted_;
    };
    struct FilteredValues {
        Matrix values;
        int vertices = 0;
        std::string mode = "direct";
    };
    FilteredValues lattice_transform(const Matrix& sources, const Matrix& queries,
    const Matrix& values, double sigma2, Backend backend,
    const FgtOptions& fgt = FgtOptions());
}
// namespace acpd
