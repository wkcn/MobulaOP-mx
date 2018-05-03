import mxnet as mx
import numpy as np
from mobula_op_mx.libs.gpu_add import add

ctx = mx.gpu(0)
dtype = np.float32

a = mx.nd.array(np.arange(5), dtype = dtype, ctx = ctx)
b = mx.nd.array(np.arange(5, 10), dtype = dtype, ctx = ctx)

c = add(a, b)
rc = a + b
print (c.asnumpy(), rc.asnumpy())
assert (c.asnumpy() == rc.asnumpy()).all()
