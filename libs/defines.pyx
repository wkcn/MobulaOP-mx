from mxnet.base import _LIB
import ctypes
cdef float * Pointer(v):
    cdef unsigned long long p
    cp = ctypes.c_void_p() 
    rtn =  _LIB.MXNDArrayGetData(v.handle, ctypes.byref(cp))
    assert rtn == 0, "Error Code {}".format(rtn)
    p = cp.value
    return <float*>p

