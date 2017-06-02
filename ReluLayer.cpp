/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#include <armadillo>
#include "Layer.h"
#include "ReluLayer.h"
#include "common.h"
#include "Blob.h"

using std::cout;
using std::endl;

namespace fn {

template <typename Dtype>
ReluLayer<Dtype>::ReluLayer()
{
}

template <typename Dtype>
ReluLayer<Dtype>::~ReluLayer()
{
}

template<typename Dtype>
void ReluLayer<Dtype>::Forward(const arma::Cube<Dtype> &bottom,
	arma::Cube<Dtype> &top) {
	if (bottom.n_rows > 0
		&& bottom.n_cols > 0
		&& bottom.n_slices > 0) {
		for (int i = 0; i < bottom.n_elem; ++i) {
			top.at(i) = std::max<Dtype>(bottom.at(i), 0);
		}
	}
}

//Explicit instantiation
INSTANTIATE_CLASS(ReluLayer);

}