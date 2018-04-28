#include "defines.hpp"

void roi_align_op_forward(const int nthreads, const float* bottom_data, const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int sampling_ratio, const float* bottom_rois, float* top_data, const int device_id);
void roi_align_op_backward(const int nthreads, const float* top_diff, const int num_rois, const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int sampling_ratio, float* bottom_diff, const float* bottom_rois, const int device_id);
