include "../pyx_def.pyx"
import mxnet as mx

cdef extern from "roi_align_op.hpp":
    void roi_align_op_forward(const int nthreads, const float* bottom_data, const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int sampling_ratio, const float* bottom_rois, float* top_data, const int device_id);
    void roi_align_op_backward(const int nthreads, const float* top_diff, const int num_rois, const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int sampling_ratio, float* bottom_diff, const float* bottom_rois, const int device_id);

def forward(data, rois, pooled_size, spatial_scale, sampling_ratio, out):
    device_id = data.context.device_id 
    roi_align_op_forward(out.size, Pointer(data), spatial_scale, data.shape[1], data.shape[2], data.shape[3], pooled_size[0], pooled_size[1], sampling_ratio, Pointer(rois), Pointer(out), device_id)

def backward(data, rois, top_diff, pooled_size, spatial_scale, sampling_ratio, data_diff):
    device_id = data.context.device_id 
    roi_align_op_backward(top_diff.size, Pointer(top_diff), rois.shape[0], spatial_scale, data.shape[1], data.shape[2], data.shape[3], pooled_size[0], pooled_size[1], sampling_ratio, Pointer(data_diff), Pointer(rois), device_id)
