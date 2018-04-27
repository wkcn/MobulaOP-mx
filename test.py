import mxnet as mx
from gpu_add import gpu_add

ctx = mx.gpu(0)
shape = (1,2,3,4)

a = mx.nd.random(shape, ctx = ctx)
b = mx.nd.random(shape, ctx = ctx)
c = gpu_add(a, b)
rc = a + b
assert (c.asnumpy() == rc.asnumpy()).all()
