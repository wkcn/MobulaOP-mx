import mxnet as mx
import numpy as np
import roi_align_op

ctx = mx.gpu(0)
dtype = np.float32

N, C, H, W = 2, 3, 4, 4

data = mx.nd.array(np.arange(N*C*H*W).reshape((N,C,H,W)), ctx = ctx, dtype = dtype)
rois = mx.nd.array([[0, 1, 1, 3, 3]], ctx = ctx, dtype = dtype)

output = mx.nd.Custom(op_type = 'ROIAlign', data = data, rois = rois, pooled_size = (2,2), spatial_scale = 1.0, sampling_ratio = 1.0)

print (output)
