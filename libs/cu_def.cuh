#include <iostream>

#define MOBULA_KERNEL __global__ void 
#define CUDA_NUM_THREADS 512
#define CUDA_GET_BLOCKS(n) ((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS

#define KERNEL_LOOP(i,n) for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < (n);i += blockDim.x * gridDim.x)


#define CUDA_1D_KERNEL_LOOP(i, n) KERNEL_LOOP(i, n)

#define KERNEL_RUN(a, n) (a)<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__ float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

inline void set_device(int device_id) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device != device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
    }
}


template <typename T>
__global__ void set_kernel(const int n, const T value, T* y);

template <typename T>
void gpu_set_value(const int n, const T value, T* y);
