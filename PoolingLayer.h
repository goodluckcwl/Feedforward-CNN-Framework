/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:56:01
*/
#ifndef FN_POOLING_LAYER_H_
#define FN_POOLING_LAYER_H_
#include "Blob.h"
#include "Layer.h"
#include <armadillo>

namespace fn {

template <typename Dtype>
class PoolingLayer:
	public Layer<Dtype>
{

public:
	PoolingLayer();
	~PoolingLayer();
	void LayerSetUp(
		const int kernel_h,const int kernel_w,
		const int stride_h, const int stride_w,
		const int pad_h, const int pad_w);

	void Forward(const arma::Cube<Dtype> &bottom,
		arma::Cube<Dtype> &top);

	void CalShape(const arma::Cube<Dtype> &bottom,
		std::vector<int> &shape);

private:
	int kernel_h_;
	int kernel_w_;
	int stride_h_;
	int stride_w_;
	int pad_h_;
	int pad_w_;
	int channels_;
	int height_;
	int width_;
	int output_h_;
	int output_w_;
	
};

}

#endif // FN_POOLING_LAYER_H_