include "../pyx_def.pyx"
import mxnet as mx

cdef extern from "roi_align_op.hpp":
    void roi_align_op_forward(const int nthreads, const float* bottom_data, const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const float* bottom_rois, float* top_data, float *argmax_x, float *argmax_y, const int device_id);
    void roi_align_op_backward(const int nthreads, const float* top_diff, const int num_rois, const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, float* bottom_diff, const float* bottom_rois, const float *argmax_x, const float *argmax_y, const int device_id);

def forward(data, rois, pooled_size, spatial_scale, out, argmax_x, argmax_y):
    device_id = data.context.device_id 
    roi_align_op_forward(out.size, Pointer(data), spatial_scale, data.shape[1], data.shape[2], data.shape[3], pooled_size[0], pooled_size[1], Pointer(rois), Pointer(out), Pointer(argmax_x), Pointer(argmax_y), device_id)

def backward(data, rois, top_diff, pooled_size, spatial_scale, data_diff, argmax_x, argmax_y):
    device_id = data.context.device_id 
    roi_align_op_backward(data_diff.size, Pointer(top_diff), rois.shape[0], spatial_scale, data.shape[1], data.shape[2], data.shape[3], pooled_size[0], pooled_size[1], Pointer(data_diff), Pointer(rois), Pointer(argmax_x), Pointer(argmax_y), device_id)
