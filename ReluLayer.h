/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:56:12
*/
#ifndef FN_RELU_LAYER_H_
#define FN_RELU_LAYER_H_

#include "Blob.h"
#include "Layer.h"
#include <armadillo>

namespace fn{

template <typename Dtype>
class ReluLayer:
	public Layer<Dtype>
{
public:
	ReluLayer();
	~ReluLayer();

	void Forward(const arma::Cube<Dtype> &bottom,
		arma::Cube<Dtype> &top);
};

}

#endif // !FN_RELU_LAYER_H_
