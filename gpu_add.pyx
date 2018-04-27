import mxnet as mx

cdef extern from "gpu_add.hpp":
    void gpu_add(const float *a, const float *b, int n, float *c, int device_id);

def gpu_add(a, b):
    assert a.shape == b.shape
    c = mx.empty(a.shape, dtype = a.dtype, ctx = a.context)
    P = lambda v : v.handle.value
    device_id = a.context.device_id 
    gpu_add(P(a), P(b), a.size, P(c), device_id)
    return c
