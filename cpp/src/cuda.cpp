#include "acpd/cuda.hpp"
#include <cmath>
#ifdef ACPD_ENABLE_CUDA
extern "C" int acpd_cuda_available(void);
extern "C" int acpd_cuda_device_name(char* buffer, int capacity);
extern "C" int acpd_cuda_gaussian_transform(const double* sources, int source_count,
const double* queries, int query_count, int dimension, const double* values,
int channels, double sigma2, int single_precision, double* out);
#endif
namespace acpd {
    void CudaOptions::validate() const {
        // Only a precision selector today; kept as a struct so device options can be
        // added without changing every call site.
    }
#ifdef ACPD_ENABLE_CUDA
    bool cuda_available() {
        return acpd_cuda_available()!=0;
    }
    std::string cuda_device_name() {
        char buffer[256]={};
        return acpd_cuda_device_name(buffer,static_cast<int>(sizeof(buffer)))>0?std::string(buffer):std::string();
    }
    Matrix cuda_gaussian_transform(const Matrix& s,const Matrix& q,const Matrix& v,double sigma2,
    const CudaOptions& options) {
        options.validate();
        if(s.rows()==0||q.rows()==0||s.cols()!=q.cols()||s.rows()!=v.rows()||v.cols()<1
        ||!s.allFinite()||!q.allFinite()||!v.allFinite()||!std::isfinite(sigma2)||sigma2<=0)
        throw std::invalid_argument("invalid CUDA Gaussian transform arguments");
        if(!cuda_available()) throw NumericalError("no CUDA device is visible; select a CPU backend explicitly");
        using RowMatrix=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        const RowMatrix sources=s,queries=q,values=v;
        RowMatrix out(q.rows(),v.cols());
        const int code=acpd_cuda_gaussian_transform(sources.data(),static_cast<int>(s.rows()),
        queries.data(),static_cast<int>(q.rows()),static_cast<int>(s.cols()),
        values.data(),static_cast<int>(v.cols()),sigma2,options.single_precision?1:0,out.data());
        if(code!=0) throw NumericalError("CUDA Gaussian transform failed with code "+std::to_string(code));
        Matrix result=out;
        require_finite(result,"CUDA Gaussian transform");
        return result;
    }
#else
    bool cuda_available() {
        return false;
    }
    std::string cuda_device_name() {
        return {};
    }
    Matrix cuda_gaussian_transform(const Matrix&,const Matrix&,const Matrix&,double,const CudaOptions&) {
        throw std::invalid_argument("this build has no CUDA support; configure with -DACPD_ENABLE_CUDA=ON");
    }
#endif
}
// namespace acpd
