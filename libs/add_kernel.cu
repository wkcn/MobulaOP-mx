#include <iostream>
#include "gpu_add.hpp"

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
