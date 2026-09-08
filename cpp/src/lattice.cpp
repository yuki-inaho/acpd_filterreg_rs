// Algorithmic port of the BSD-3-Clause permutohedral code shipped in probreg.
// Copyright (c) 2013 Philipp Kraehenbuehl. See THIRD_PARTY_NOTICES.md.
#include "acpd/lattice.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
namespace acpd {
    std::size_t LatticeKeyHash::operator()(const LatticeKey& key) const noexcept {
        std::size_t hash = 0;
        for (auto value : key) {
            hash += static_cast<std::size_t>(value);
            hash *= 1664525u;
        }
        return hash;
    }
    namespace {
        void check_features(const Matrix& f) {
            if (f.rows()==0 || f.cols()<1 || f.cols()>16 || f.rows()>std::numeric_limits<int>::max()/17 || !f.allFinite())
            throw std::invalid_argument("lattice features need finite shape (n,1..16), n>0");
        }
    }
    void enclosing_simplex(const Eigen::Ref<const Eigen::RowVectorXd>& f, bool blur, Simplex& out) {
        const int d=static_cast<int>(f.size()), width=d+1;
        if (d<1 || d>max_lattice_dimension || !f.allFinite()) throw std::invalid_argument("invalid lattice feature");
        // Fixed-capacity scratch: d is bounded by max_lattice_dimension, so width<=17.
        std::array<double,max_lattice_dimension+1> elevated{},rem{};
        std::array<double,max_lattice_dimension+2> bary{};
        std::array<int,max_lattice_dimension+1> rank{};
        // Blur changes the variance of the lattice kernel. These two constants are
        // NOT interchangeable: sqrt(2/3)*(d+1) vs sqrt(1/6)*(d+1).
        const double inv_std=std::sqrt(blur ? 2.0/3.0 : 1.0/6.0)*width;
        double sum_feature=0;
        for(int j=d;j>0;--j) {
            const double scaled=f[j-1]*inv_std/std::sqrt(double(j*(j+1)));
            elevated[j]=sum_feature-j*scaled;
            sum_feature+=scaled;
        }
        elevated[0]=sum_feature;
        std::int64_t sum=0;
        for(int i=0;i<width;++i) {
            // Bounded before float->integer conversion; widened keys support far
            // larger normalized coordinates than the upstream signed-short keys.
            if(!std::isfinite(elevated[i]) || std::abs(elevated[i])>0x1p46)
            throw NumericalError("lattice coordinate overflow: rescale points or increase sigma2");
            const double v=elevated[i]/width;
            const double upper=std::ceil(v)*width, lower=std::floor(v)*width;
            rem[i]=(upper-elevated[i] < elevated[i]-lower) ? upper : lower;
            sum+=static_cast<std::int64_t>(rem[i]);
        }
        sum/=width;
        for(int i=0;i<d;++i) for(int j=i+1;j<width;++j) {
            if(elevated[i]-rem[i] < elevated[j]-rem[j]) ++rank[i];
            else ++rank[j];
        }
        for(int i=0;i<width;++i) {
            rank[i]+=static_cast<int>(sum);
            if(rank[i]<0) {
                rank[i]+=width;
                rem[i]+=width;
            }
            else if(rank[i]>d) {
                rank[i]-=width;
                rem[i]-=width;
            }
            if(rank[i]<0 || rank[i]>d) throw NumericalError("invalid simplex rank");
        }
        for(int i=0;i<width;++i) {
            const double delta=(elevated[i]-rem[i])/width;
            bary[d-rank[i]]+=delta;
            bary[d-rank[i]+1]-=delta;
        }
        bary[0]+=1+bary[width];
        out.keys.assign(static_cast<std::size_t>(width),LatticeKey(d));
        out.weights.resize(static_cast<std::size_t>(width));
        for(int color=0;color<width;++color) {
            for(int axis=0;axis<d;++axis)
            out.keys[color][axis]=static_cast<std::int64_t>(rem[axis])+color-(rank[axis]>d-color?width:0);
            out.weights[color]=bary[color];
            if(!std::isfinite(bary[color]) || bary[color]<-1e-8)
            throw NumericalError("invalid barycentric weight");
        }
    }
    Simplex enclosing_simplex(const Eigen::Ref<const Eigen::RowVectorXd>& f, bool blur) {
        Simplex out;
        enclosing_simplex(f,blur,out);
        return out;
    }
    Permutohedral::Permutohedral(const Matrix& features, bool blur)
    : n_(static_cast<int>(features.rows())),d_(static_cast<int>(features.cols())),with_blur_(blur) {
        check_features(features);
        std::unordered_map<LatticeKey,std::size_t,LatticeKeyHash> index;
        index.reserve(static_cast<std::size_t>(n_)*(d_+1));
        offsets_.reserve(static_cast<std::size_t>(n_)*(d_+1));
        barycentric_.reserve(offsets_.capacity());
        Simplex simplex;
        for(int i=0;i<n_;++i) {
            enclosing_simplex(features.row(i),blur,simplex);
            for(int j=0;j<=d_;++j) {
                auto inserted=index.emplace(simplex.keys[j],keys_.size());
                if(inserted.second) keys_.push_back(simplex.keys[j]);
                offsets_.push_back(inserted.first->second);
                barycentric_.push_back(simplex.weights[j]);
            }
        }
        if(keys_.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()-1))
        throw NumericalError("too many lattice vertices");
        if(blur) {
            neighbors_.reserve((d_+1)*keys_.size());
            for(int axis=0;axis<=d_;++axis) for(const auto& key:keys_) {
                LatticeKey left=key,right=key;
                for(int k=0;k<d_;++k) {
                    --left[k];
                    ++right[k];
                }
                // The final elevated coordinate is redundant and is not stored.
                // In particular, never read key[d] (an upstream scalar edge defect).
                if(axis<d_) {
                    left[axis]=key[axis]+d_;
                    right[axis]=key[axis]-d_;
                }
                auto l=index.find(left),r=index.find(right);
                neighbors_.emplace_back(l==index.end()?-1:static_cast<int>(l->second),
                r==index.end()?-1:static_cast<int>(r->second));
            }
        }
    }
    Matrix Permutohedral::filter(const Matrix& values,int start,bool reverse) const {
        if(values.rows()!=n_ || values.cols()<1 || !values.allFinite() || start<0 || start>n_)
        throw std::invalid_argument("invalid lattice values or splat start");
        const auto count=static_cast<int>(keys_.size());
        // Row-major accumulators and inputs: every operation below is one whole row,
        // so this only changes memory layout, not the arithmetic or its order.
        using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        const RowMatrix v=values;
        RowMatrix a=RowMatrix::Zero(count,values.cols()),b=RowMatrix::Zero(count,values.cols());
        for(int i=start;i<n_;++i) for(int j=0;j<=d_;++j) {
            const std::size_t k=static_cast<std::size_t>(i)*(d_+1)+j;
            a.row(offsets_[k])+=barycentric_[k]*v.row(i);
        }
        if(with_blur_) for(int pass=0;pass<=d_;++pass) {
            const int axis=reverse?d_-pass:pass;
            for(int i=0;i<count;++i) {
                const auto neighbor=neighbors_[static_cast<std::size_t>(axis)*count+i];
                b.row(i)=a.row(i);
                if(neighbor.first>=0) b.row(i)+=0.5*a.row(neighbor.first);
                if(neighbor.second>=0) b.row(i)+=0.5*a.row(neighbor.second);
            }
            a.swap(b);
        }
        const double gain=1.0/(1.0+std::ldexp(1.0,-d_));
        RowMatrix sliced=RowMatrix::Zero(n_,values.cols());
        for(int i=0;i<n_;++i) for(int j=0;j<=d_;++j) {
            const std::size_t k=static_cast<std::size_t>(i)*(d_+1)+j;
            sliced.row(i)+=(gain*barycentric_[k])*a.row(offsets_[k]);
        }
        Matrix out=sliced;
        require_finite(out,"lattice filter");
        return out;
    }
    FixedNoBlurLattice::FixedNoBlurLattice(const Matrix& f,const Matrix& v):d_(static_cast<int>(f.cols())) {
        check_features(f);
        if(v.rows()!=f.rows() || v.cols()<1 || !v.allFinite()) throw std::invalid_argument("invalid no-blur values");
        std::vector<std::size_t> offsets;
        std::vector<double> weights;
        index_.reserve(static_cast<std::size_t>(f.rows())*(d_+1));
        Simplex s;
        for(int i=0;i<f.rows();++i) {
            enclosing_simplex(f.row(i),false,s);
            for(int j=0;j<=d_;++j) {
                auto item=index_.emplace(s.keys[j],keys_.size());
                if(item.second) keys_.push_back(s.keys[j]);
                offsets.push_back(item.first->second);
                weights.push_back(s.weights[j]);
            }
        }
        splatted_.setZero(static_cast<int>(keys_.size()),v.cols());
        const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> rows=v;
        for(int i=0;i<f.rows();++i) for(int j=0;j<=d_;++j) {
            const auto k=static_cast<std::size_t>(i)*(d_+1)+j;
            splatted_.row(offsets[k])+=weights[k]*rows.row(i);
        }
    }
    Matrix FixedNoBlurLattice::slice(const Matrix& queries) const {
        check_features(queries);
        if(queries.cols()!=d_) throw std::invalid_argument("no-blur feature dimension mismatch");
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> sliced
        =Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::Zero(queries.rows(),splatted_.cols());
        Simplex s;
        for(int i=0;i<queries.rows();++i) {
            enclosing_simplex(queries.row(i),false,s);
            for(int j=0;j<=d_;++j) {
                const auto found=index_.find(s.keys[j]);
                if(found!=index_.end()) sliced.row(i)+=s.weights[j]*splatted_.row(found->second);
            }
        }
        Matrix out=sliced;
        require_finite(out,"no-blur slicing");
        return out;
    }
    FilteredValues lattice_transform(const Matrix& s,const Matrix& q,const Matrix& v,double sigma2,Backend backend) {
        check_features(s);
        check_features(q);
        if(s.cols()!=q.cols() || s.rows()!=v.rows() || v.cols()<1 || !v.allFinite() || !std::isfinite(sigma2) || sigma2<=0)
        throw std::invalid_argument("invalid Gaussian transform arguments");
        if(backend==Backend::Direct) {
            Matrix out=Matrix::Zero(q.rows(),v.cols());
            for(int i=0;i<q.rows();++i) for(int j=0;j<s.rows();++j)
            out.row(i)+=std::exp(-0.5*(q.row(i)-s.row(j)).squaredNorm()/sigma2)*v.row(j);
            require_finite(out,"direct Gaussian transform");
            return {
                out,0,"direct"
            };
        }
        const double sigma=std::sqrt(sigma2);
        if(backend==Backend::PermutohedralNoBlur) {
            FixedNoBlurLattice lattice(s/sigma,v);
            return {
                lattice.slice(q/sigma),static_cast<int>(lattice.lattice_size()),"original_noblur"
            };
        }
        Matrix f(q.rows()+s.rows(),s.cols()),values=Matrix::Zero(q.rows()+s.rows(),v.cols());
        f.topRows(q.rows())=q/sigma;
        f.bottomRows(s.rows())=s/sigma;
        values.bottomRows(s.rows())=v;
        Permutohedral lattice(f,true);
        if(backend==Backend::Probreg && lattice.lattice_size()>0.015*s.rows()) {
            Permutohedral no_blur(f,false);
            return {
                no_blur.filter(values,static_cast<int>(q.rows())).topRows(q.rows()),
                static_cast<int>(no_blur.lattice_size()),"probreg_noblur"
            };
        }
        return {
            lattice.filter(values,static_cast<int>(q.rows())).topRows(q.rows()),
            static_cast<int>(lattice.lattice_size()),"blur"
        };
    }
}
// namespace acpd
