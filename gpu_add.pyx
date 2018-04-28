import mxnet as mx
from mxnet.base import _LIB
import ctypes

cdef extern from "gpu_add.hpp":
    void gpu_add(const float *a, const float *b, int n, float *c, int device_id);

cdef float * Pointer(v):
    cdef unsigned long long p
    cp = ctypes.c_void_p() 
    rtn =  _LIB.MXNDArrayGetData(v.handle, ctypes.byref(cp))
    assert rtn == 0, "Error Code {}".format(rtn)
    p = cp.value
    return <float*>p

def add(a, b):
    assert a.shape == b.shape
    c = mx.nd.full(val = 3939, shape = a.shape, dtype = a.dtype, ctx = a.context)
    device_id = a.context.device_id 
    gpu_add(Pointer(a), Pointer(b), a.size, Pointer(c), device_id)
    return c
