#include "gpu_add.hpp"

#define MOBULA_KERNEL __global__ void 
#define CUDA_NUM_THREADS 512
#define CUDA_GET_BLOCKS(n) ((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS

#define KERNEL_LOOP(i,n) for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < (n);i += blockDim.x * gridDim.x)

#define KERNEL_RUN(a, n) (a)<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>


MOBULA_KERNEL add_kernel(const int n, const float *a, const float *b, float *output){
	KERNEL_LOOP(i, n){
		output[i] = a[i] + b[i];
	}
}

void set_device(int device_id) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device != device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
    }
}

void gpu_add(const float *a, const float *b, int n, float *c, int device_id) {
    set_device(device_id);
    KERNEL_RUN(add_kernel, n)(n, a, b, c);
}
