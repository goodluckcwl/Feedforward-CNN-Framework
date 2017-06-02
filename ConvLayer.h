/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:54:41
*/
#ifndef FN_CONV_LAYER_H_
#define FN_CONV_LAYER_H_

#include <vector>
#include "Layer.h"

namespace fn {

template <typename Dtype>
class ConvLayer :
	public Layer<Dtype>
{
public:
	ConvLayer();
	~ConvLayer();
	/**
	 * [LayerSetUp description]
	 * @Author   Weiliang                 Chen
	 * @DateTime 2016-09-29T10:32:36+0800
	 * @param    weights                  [description]
	 * @param    bias                     [description]
	 * @param    stride_h                 [description]
	 * @param    stride_w                 [description]
	 * @param    pad_h                    [description]
	 * @param    pad_w                    [description]
	 */
	void LayerSetUp(const Blob<Dtype> &weights, const arma::Col<Dtype> &bias,
		const int stride_h, const int stride_w,
		const int pad_h, const int pad_w);

	void Forward(const std::vector<arma::Cube<Dtype>*> &bottom,
		std::vector<arma::Cube<Dtype>*> &top);

	void Forward(const arma::Cube<Dtype> &bottom,
		arma::Cube<Dtype> &top);

	void CalShape(const arma::Cube<Dtype> &bottom, 
		std::vector<int> &shape);

private:
	Blob<Dtype> weights_;
	arma::Col<Dtype> bias_;
	int kernel_h_;
	int kernel_w_;
	int channels_;
	int number_;
	int stride_h_;
	int stride_w_;
	int pad_h_;
	int pad_w_;
};



}	// namespace fn

#endif	// !FN_CONV_LAYER_H_
