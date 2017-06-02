/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#ifndef FN_INNER_PRODUCT_LAYER_H_
#define FN_INNER_PRODUCT_LAYER_H_

#include "Layer.h"


namespace fn {

template <typename Dtype>
class InnerProductLayer:
	public Layer<Dtype>
{
public:
	InnerProductLayer();
	~InnerProductLayer();
	void LayerSetUp(const arma::Mat<Dtype> &weights, const arma::Col<Dtype> &bias);
	void Forward(const arma::Cube<Dtype> &bottom,
		arma::Col<Dtype> &top);
	void CalShape(const arma::Cube<Dtype> &bottom,
		std::vector<int> &shape);
private:
	arma::Mat<Dtype> weights_;
	arma::Col<Dtype> bias_;
};



}

#endif // FN_INNER_PRODUCT_LAYER_H_

