// CUDA Gaussian-sum operator for the E-step.
//
// Both directions of the E-step reduce to one operator,
//
//     out[q, ch] = sum_s exp(-|q - s|^2 / (2 sigma^2)) * value[s, ch],
//
// evaluated once for the FilterReg direction (sources = fixed, carrying the
// [1, x, |x|^2] moment channels) and twice for the CPD direction (a forward pass
// with unit values for the denominators, then a transpose pass with the weighted
// moments). That is the decomposition of the GPU-oriented reformulation: the
// N x M correspondence matrix is never formed, only the operator's input and
// output arrays cross the boundary.
//
// The interface is deliberately a plain C ABI over raw doubles, so this
// translation unit can be compiled by nvcc's supported host compiler while the
// rest of the core is built by a newer one.
//
// Precision: this is exact pair evaluation, not an approximation - the only
// approximation available here is `single_precision`, which computes the
// exponential and the products in float while accumulating in double. On a
// Pascal consumer part FP64 runs at 1/32 rate, so that switch is the difference
// between a compute-bound kernel and a crawl; it is never selected implicitly.
#include <cuda_runtime.h>
#include <cstddef>

namespace {

constexpr int kMaxChannels = 8;   // d + 2 with normals is 8 at d = 3
constexpr int kMaxDimension = 16;
constexpr int kGeometryDimension = 3;
constexpr int kTile = 128;
constexpr int kBlock = 128;

template <typename Compute, int MaxDimension>
__global__ void gaussian_transform_kernel(const double* __restrict__ sources,
                                          const double* __restrict__ values,
                                          const double* __restrict__ queries,
                                          double* __restrict__ out,
                                          int source_count, int query_count,
                                          int dimension, int channels,
                                          double inverse_two_sigma_squared) {
    __shared__ double tile_source[kTile * MaxDimension];
    __shared__ double tile_value[kTile * kMaxChannels];

    const int query = blockIdx.x * blockDim.x + threadIdx.x;
    double position[MaxDimension];
    double accumulator[kMaxChannels];
    for (int c = 0; c < channels; ++c) accumulator[c] = 0.0;
    if (query < query_count) {
        for (int a = 0; a < dimension; ++a) {
            position[a] = queries[static_cast<std::size_t>(query) * dimension + a];
        }
    }

    for (int base = 0; base < source_count; base += kTile) {
        const int span = min(kTile, source_count - base);
        for (int k = threadIdx.x; k < span; k += blockDim.x) {
            const std::size_t source = static_cast<std::size_t>(base + k);
            for (int a = 0; a < dimension; ++a) {
                tile_source[k * MaxDimension + a] = sources[source * dimension + a];
            }
            for (int c = 0; c < channels; ++c) {
                tile_value[k * kMaxChannels + c] = values[source * channels + c];
            }
        }
        __syncthreads();
        if (query < query_count) {
            for (int k = 0; k < span; ++k) {
                Compute distance = Compute(0);
                for (int a = 0; a < dimension; ++a) {
                    const Compute delta =
                        static_cast<Compute>(position[a]) -
                        static_cast<Compute>(tile_source[k * MaxDimension + a]);
                    distance += delta * delta;
                }
                const Compute weight =
                    exp(-distance * static_cast<Compute>(inverse_two_sigma_squared));
                // Accumulation stays in double whatever the evaluation precision is.
                for (int c = 0; c < channels; ++c) {
                    accumulator[c] += static_cast<double>(weight) *
                                      tile_value[k * kMaxChannels + c];
                }
            }
        }
        __syncthreads();
    }

    if (query < query_count) {
        for (int c = 0; c < channels; ++c) {
            out[static_cast<std::size_t>(query) * channels + c] = accumulator[c];
        }
    }
}

int device_count() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

}  // namespace

extern "C" int acpd_cuda_available(void) { return device_count() > 0 ? 1 : 0; }

extern "C" int acpd_cuda_device_name(char* buffer, int capacity) {
    if (buffer == nullptr || capacity <= 0 || device_count() == 0) return 0;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 0;
    int written = 0;
    while (written < capacity - 1 && properties.name[written] != '\0') {
        buffer[written] = properties.name[written];
        ++written;
    }
    buffer[written] = '\0';
    return written;
}

// Returns 0 on success, a negative code on failure. The caller reports the code;
// this layer never substitutes a CPU result for a failed device computation.
extern "C" int acpd_cuda_gaussian_transform(const double* sources, int source_count,
                                            const double* queries, int query_count,
                                            int dimension, const double* values,
                                            int channels, double sigma2,
                                            int single_precision, double* out) {
    if (sources == nullptr || queries == nullptr || values == nullptr || out == nullptr) return -1;
    if (source_count <= 0 || query_count <= 0) return -2;
    if (dimension < 1 || dimension > kMaxDimension) return -3;
    if (channels < 1 || channels > kMaxChannels) return -4;
    if (!(sigma2 > 0)) return -5;
    if (device_count() == 0) return -6;

    const std::size_t source_bytes = static_cast<std::size_t>(source_count) * dimension * sizeof(double);
    const std::size_t value_bytes = static_cast<std::size_t>(source_count) * channels * sizeof(double);
    const std::size_t query_bytes = static_cast<std::size_t>(query_count) * dimension * sizeof(double);
    const std::size_t out_bytes = static_cast<std::size_t>(query_count) * channels * sizeof(double);

    double *device_sources = nullptr, *device_values = nullptr;
    double *device_queries = nullptr, *device_out = nullptr;
    int status = 0;
    auto release = [&]() {
        cudaFree(device_sources);
        cudaFree(device_values);
        cudaFree(device_queries);
        cudaFree(device_out);
    };
    if (cudaMalloc(&device_sources, source_bytes) != cudaSuccess ||
        cudaMalloc(&device_values, value_bytes) != cudaSuccess ||
        cudaMalloc(&device_queries, query_bytes) != cudaSuccess ||
        cudaMalloc(&device_out, out_bytes) != cudaSuccess) {
        release();
        return -7;
    }
    if (cudaMemcpy(device_sources, sources, source_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_values, values, value_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_queries, queries, query_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        release();
        return -8;
    }

    const double inverse_two_sigma_squared = 0.5 / sigma2;
    const int blocks = (query_count + kBlock - 1) / kBlock;
    if (single_precision && dimension <= kGeometryDimension) {
        gaussian_transform_kernel<float, kGeometryDimension><<<blocks, kBlock>>>(
            device_sources, device_values, device_queries, device_out, source_count,
            query_count, dimension, channels, inverse_two_sigma_squared);
    } else if (single_precision) {
        gaussian_transform_kernel<float, kMaxDimension><<<blocks, kBlock>>>(
            device_sources, device_values, device_queries, device_out, source_count,
            query_count, dimension, channels, inverse_two_sigma_squared);
    } else if (dimension <= kGeometryDimension) {
        gaussian_transform_kernel<double, kGeometryDimension><<<blocks, kBlock>>>(
            device_sources, device_values, device_queries, device_out, source_count,
            query_count, dimension, channels, inverse_two_sigma_squared);
    } else {
        gaussian_transform_kernel<double, kMaxDimension><<<blocks, kBlock>>>(
            device_sources, device_values, device_queries, device_out, source_count,
            query_count, dimension, channels, inverse_two_sigma_squared);
    }
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        release();
        return -9;
    }
    if (cudaMemcpy(out, device_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) status = -10;
    release();
    return status;
}
