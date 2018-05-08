/*!
 * Copyright (c) 2017 by Contributors
 * \file roi_align.cu
 * \brief roi align operator
 * \author Yuchen Guo, Zehao Shi
*/
#include "roi_align_op.hpp"
#include "../cu_def.cuh"

template<typename Dtype>
__global__ void ROIAlignForwardKernel(const int count, const Dtype* bottom_data,
                                     const float spatial_scale,
                                     const int channels, const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const Dtype* bottom_rois, Dtype* top_data,
                                     Dtype* argmax_x, Dtype* argmax_y) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      argmax_x[index] = 0;
      argmax_y[index] = 0;
      continue;
    }

    Dtype roi_start_w = (bottom_rois[1]) * spatial_scale;
    Dtype roi_start_h = (bottom_rois[2]) * spatial_scale;
    Dtype roi_end_w = (bottom_rois[3]) * spatial_scale;
    Dtype roi_end_h = (bottom_rois[4]) * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(1));
    Dtype roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(1));
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>((ph) * bin_size_h);
    Dtype wstart = static_cast<Dtype>((pw) * bin_size_w);
    Dtype hend = static_cast<Dtype>((ph + 1) * bin_size_h);
    Dtype wend = static_cast<Dtype>((pw + 1) * bin_size_w);

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, static_cast<Dtype>(0)), static_cast<Dtype>(height));
    hend = min(max(hend + roi_start_h, static_cast<Dtype>(0)), static_cast<Dtype>(height));
    wstart = min(max(wstart + roi_start_w, static_cast<Dtype>(0)), static_cast<Dtype>(width));
    wend = min(max(wend + roi_start_w, static_cast<Dtype>(0)), static_cast<Dtype>(width));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    Dtype maxidx_x = -1;
    Dtype maxidx_y = -1;

    bottom_data += (roi_batch_ind * channels + c) * height * width;
    Dtype h_stride = (hend - hstart)/3.0;
    Dtype w_stride = (wend - wstart)/3.0;
    for (Dtype h = hstart+h_stride; h <= hend-h_stride+0.01; h += max(h_stride, 0.01)) {
      for (Dtype w = wstart+w_stride; w <= wend-w_stride+0.01; w += max(w_stride, 0.01)) {
        int hlow = min(max(static_cast<int>(floor(h)), 0), height-1);
        int hhigh = min(max(static_cast<int>(ceil(h)), 0), height-1);
        int wleft = min(max(static_cast<int>(floor(w)), 0), width-1);
        int wright = min(max(static_cast<int>(ceil(w)), 0), width-1);
        int topleft = hlow * width + wleft;
        int topright = hlow * width + wright;
        int bottomleft = hhigh * width + wleft;
        int bottomright = hhigh * width + wright;
        
        Dtype alpha = (hlow == hhigh) ? static_cast<Dtype>(0.5) : (h - hlow) / (hhigh - hlow);
        Dtype beta = (wleft == wright) ? static_cast<Dtype>(0.5) : (w - wleft) / (wright - wleft);
        Dtype value = (1 - alpha) * (1 - beta) * bottom_data[topleft] + alpha * (1 - beta) * bottom_data[bottomleft]
                            + (1 - alpha) * beta * bottom_data[topright] + alpha * beta * bottom_data[bottomright];
        
        if (value > maxval) {
          maxval = value;
          maxidx_x = w;
          maxidx_y = h;
        }
      }
    }
    top_data[index] = maxval;
    argmax_x[index] = (Dtype)maxidx_x;
    argmax_y[index] = (Dtype)maxidx_y;
  }
}

template<typename Dtype>
__global__ void ROIAlignBackwardAccKernel(const int count, const Dtype* top_diff,
                                         const Dtype* argmax_x, const Dtype* argmax_y, 
                                         const int num_rois,
                                         const float spatial_scale,
                                         const int channels, const int height, const int width,
                                         const int pooled_height, const int pooled_width,
                                         Dtype* bottom_diff, const Dtype* bottom_rois) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      Dtype roi_start_w = (offset_bottom_rois[1]) * spatial_scale;
      Dtype roi_start_h = (offset_bottom_rois[2]) * spatial_scale;
      Dtype roi_end_w = (offset_bottom_rois[3]) * spatial_scale;
      Dtype roi_end_h = (offset_bottom_rois[4]) * spatial_scale;

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w > roi_start_w - 1.0 && w < roi_end_w + 1.0 &&
                           h > roi_start_h - 1.0 && h < roi_end_h + 1.0);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_x = argmax_x + offset;
      const Dtype* offset_argmax_y = argmax_y + offset;

      // Force malformed ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(1));
      Dtype roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(1));

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          const int pool_index = ph * pooled_width + pw;
          Dtype a_x = offset_argmax_x[pool_index];
          Dtype a_y = offset_argmax_y[pool_index];
          int hlow = min(max(static_cast<int>(floor(a_y)), 0), height-1);
          int hhigh = min(max(static_cast<int>(ceil(a_y)), 0), height-1);
          int wleft = min(max(static_cast<int>(floor(a_x)), 0), width-1);
          int wright = min(max(static_cast<int>(ceil(a_x)), 0), width-1);

          if (h != hlow && h != hhigh && w != wleft && w != wright) // (w, h) is not around (a_x, a_y)
              continue;
              

          Dtype alpha = (hlow == hhigh) ? static_cast<Dtype>(0.5) : (a_y - hlow) / (hhigh - hlow);
          Dtype beta = (wleft == wright) ? static_cast<Dtype>(0.5) : (a_x - wleft) / (wright - wleft);
          if (h == hlow && w == wleft) gradient += offset_top_diff[pool_index] * (1 - alpha) * (1 - beta);
          else if (h == hlow && w == wright) gradient += offset_top_diff[pool_index] * (1 - alpha) * beta;
          else if (h == hhigh && w == wleft) gradient += offset_top_diff[pool_index] * alpha * (1 - beta);
          else if (h == hhigh && w == wright) gradient += offset_top_diff[pool_index] * alpha * beta;
        }
      }
    }
    bottom_diff[index] += gradient;
  }
}

void roi_align_op_forward(const int nthreads, const float* bottom_data, const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const float* bottom_rois, float* top_data, float* argmax_x, float* argmax_y, const int device_id) {
    set_device(device_id);
    KERNEL_RUN(ROIAlignForwardKernel, nthreads)(nthreads, bottom_data, spatial_scale, channels, height, width, pooled_height, pooled_width, bottom_rois, top_data, argmax_x, argmax_y);
}

void roi_align_op_backward(const int nthreads, const float* top_diff, const int num_rois, const float spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, float* bottom_diff, const float* bottom_rois, const float *argmax_x, const float *argmax_y, const int device_id) {
    set_device(device_id);
    KERNEL_RUN(ROIAlignBackwardAccKernel, nthreads)(nthreads, top_diff, argmax_x, argmax_y, num_rois, spatial_scale, channels, height, width, pooled_height, pooled_width, bottom_diff, bottom_rois);
}
