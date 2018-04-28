include "pyx_def.pyx"
import mxnet as mx

cdef extern from "gpu_add.hpp":
    void gpu_add(const float *a, const float *b, int n, float *c, int device_id);

def add(a, b):
    assert a.shape == b.shape
    c = mx.nd.empty(shape = a.shape, dtype = a.dtype, ctx = a.context)
    device_id = a.context.device_id 
    gpu_add(Pointer(a), Pointer(b), a.size, Pointer(c), device_id)
    return c
