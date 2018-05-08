import mxnet as mx
import numpy as np
import mobula_op_mx.operators.roi_align_op

ctx = mx.gpu(0)
dtype = np.float32

N, C, H, W = 2, 3, 4, 4

data = np.arange(N*C*H*W).astype(dtype).reshape((N,C,H,W))
rois = np.array([[0, 1, 1, 3, 3]], dtype = dtype)

data_sym = mx.sym.Variable('data')
rois_sym = mx.sym.Variable('rois')

output_sym = mx.sym.Custom(op_type = 'ROIAlign', data = data_sym, rois = rois_sym, pooled_size = (2,2), spatial_scale = 1.0)
output_sym = mx.sym.MakeLoss(output_sym[0])

print (data.shape, rois.shape)
exe = output_sym.simple_bind(ctx, data = data.shape, rois = rois.shape) 
exe.forward(data = data, rois = rois)

res = exe.outputs[0].asnumpy()
print (res)

exe.backward()
