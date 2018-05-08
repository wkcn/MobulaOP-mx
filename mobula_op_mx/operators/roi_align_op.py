from ..libs import roi_align_op
import mxnet as mx
import numpy as np

class ROIAlignOP(mx.operator.CustomOp):
    def __init__(self, pooled_size, spatial_scale):
        super(ROIAlignOP, self).__init__() 
        self.pooled_size = pooled_size
        self.spatial_scale = spatial_scale
    def forward(self, is_train, req, in_data, out_data, aux):
        if req[0] == 'null':
            return
        data = in_data[0]
        rois = in_data[1]
        out, argmax_x, argmax_y = out_data
        if req[0] == 'add':
            out_temp = self.get_ndarray_temp(out)
            roi_align_op.forward(data, rois, self.pooled_size, self.spatial_scale, out_temp, argmax_x, argmax_y)
            out[:] += out_temp
        else:
            roi_align_op.forward(data, rois, self.pooled_size, self.spatial_scale, out, argmax_x, argmax_y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        if req[0] == 'null':
            return
        data = in_data[0]
        rois = in_data[1]
        data_diff = in_grad[0]
        top_diff = out_grad[0]
        argmax_x = out_data[1]
        argmax_y = out_data[2]
        if req[0] == 'add':
            roi_align_op.backward(data, rois, top_diff, self.pooled_size, self.spatial_scale, data_diff, argmax_x, argmax_y)
        else:
            data_diff[:] = 0
            roi_align_op.backward(data, rois, top_diff, self.pooled_size, self.spatial_scale, data_diff, argmax_x, argmax_y)
        self.assign(in_grad[1], req[1], 0)
    def get_ndarray_temp(self, out):
        return mx.nd.empty(shape = out.shape, dtype = out.dtype, ctx = out.context)

@mx.operator.register('ROIAlign')
class ROIAlignProp(mx.operator.CustomOpProp):
    def __init__(self, pooled_size, spatial_scale):
        super(ROIAlignProp, self).__init__(need_top_grad = True)
        str2tuple = lambda s, dtype : np.fromstring(s[1:-1], dtype = dtype, sep = ',')
        self.pooled_size = str2tuple(pooled_size, int)
        self.spatial_scale = float(spatial_scale)

    def list_arguments(self):
        return ['data', 'rois']

    def list_outputs(self):
        return ['output', 'argmax_x', 'argmax_y']

    def infer_shape(self, in_shape):
        dshape, rshape = in_shape
        assert len(dshape) == 4
        assert len(rshape) == 2
        assert rshape[1] == 5
        oshape = [rshape[0], dshape[1], self.pooled_size[0], self.pooled_size[1]]
        return [dshape, rshape], [oshape, oshape, oshape]

    def create_operator(self, ctx, shapes, dtypes):
        return ROIAlignOP(self.pooled_size, self.spatial_scale)
