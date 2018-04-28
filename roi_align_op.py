from .libs import roi_align_op

class ROIAlignOP(mx.operator.CustomOp):
    def __init__(self, pooled_size, spatial_scale):
        super(ROIAlignOP, self).__init__() 
        self.pooled_size = pooled_size
        self.spatial_scale = spatial_scale
