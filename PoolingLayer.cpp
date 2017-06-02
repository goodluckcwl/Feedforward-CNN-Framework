/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#include "PoolingLayer.h"
#include "Layer.h"
#include "common.h"
#include "Blob.h"

using std::cout;
using std::endl;
namespace fn {

template <typename Dtype>
PoolingLayer<Dtype>::PoolingLayer()
{
}

template <typename Dtype>
PoolingLayer<Dtype>::~PoolingLayer()
{
}

template<typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(
	const int kernel_h, const int kernel_w,
	const int stride_h, const int stride_w,
	const int pad_h, const int pad_w)
{
	kernel_h_ = kernel_h;
	kernel_w_ = kernel_w;
	stride_h_ = stride_h;
	stride_w_ = stride_w;
	pad_h_ = pad_h;
	pad_w_ = pad_w;
}



template<typename Dtype>
void PoolingLayer<Dtype>::Forward(const arma::Cube<Dtype> &bottom,
	arma::Cube<Dtype> &top) {
	output_h_ = ceil((bottom.n_rows + 2 * pad_h_ - kernel_h_)*1.0 / stride_h_ + 1);
	output_w_ = ceil((bottom.n_cols + 2 * pad_w_ - kernel_w_)*1.0 / stride_w_ + 1);
	height_ = bottom.n_rows;
	width_ = bottom.n_cols;
	channels_ = bottom.n_slices;
	Dtype *top_data = top.memptr();
	const Dtype *bottom_data = bottom.memptr();
	//Max-pooling
	for (int c = 0; c < channels_; ++c) {
		for (int ph = 0; ph < output_h_; ++ph) {
			for (int pw = 0; pw < output_w_; ++pw) {
				int hstart = ph*stride_h_ - pad_h_;
				int wstart = pw*stride_w_ - pad_w_;
				hstart = std::max(hstart, 0);
				wstart = std::max(wstart, 0);
				int hend = std::min(hstart + kernel_h_, height_);
				int wend = std::min(wstart + kernel_w_, width_);
				const int pool_index = pw*output_h_ + ph;
				top_data[pool_index] = bottom_data[wstart * height_ + hstart];
				//Find the largest element.
				for (int h = hstart; h < hend; ++h) {
					for (int w = wstart; w < wend; ++w) {
						const int index = w*height_ + h;
						//
						if (bottom_data[index] > top_data[pool_index]) {
							top_data[pool_index] = bottom_data[index];
						}
					}
				}
			}
		}
		top_data += top.n_elem_slice;
		bottom_data += bottom.n_elem_slice;

	}
}

template<typename Dtype>
void PoolingLayer<Dtype>::CalShape(const arma::Cube<Dtype>& bottom, std::vector<int>& shape)
{
	const int output_h = ceil((bottom.n_rows + 2 * pad_h_ - kernel_h_)*1.0 / stride_h_ + 1);
	const int output_w = ceil((bottom.n_cols + 2 * pad_w_ - kernel_w_)*1.0 / stride_w_ + 1);
	shape[0] = output_h;
	shape[1] = output_w;
	shape[2] = bottom.n_slices;
}



//Explicit instantiation
INSTANTIATE_CLASS(PoolingLayer);

}