/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#include "InnerProductLayer.h"
#include "common.h"
#include "Utils.h"

namespace fn {

template<typename Dtype>
InnerProductLayer<Dtype>::InnerProductLayer()
{
}

template<typename Dtype>
InnerProductLayer<Dtype>::~InnerProductLayer()
{
}

template<typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const arma::Mat<Dtype>& weights, const arma::Col<Dtype>& bias)
{
	weights_ = weights;
	bias_ = bias;
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Forward(const arma::Cube<Dtype>& bottom, arma::Col<Dtype>& top)
{
	arma::Mat<Dtype> feat_mat;
	im2col(bottom, bottom.n_rows, bottom.n_cols, 0, 0, 1, 1, feat_mat);
	arma::Mat<Dtype>result = feat_mat * weights_;
	for (int i = 0; i < result.n_elem; ++i) {
		top.at(i) = result.at(i);
	}
	top += bias_;
}

template<typename Dtype>
void InnerProductLayer<Dtype>::CalShape(const arma::Cube<Dtype>& bottom, std::vector<int>& shape)
{
	NOT_IMPLEMENTED
}

INSTANTIATE_CLASS(InnerProductLayer);

}
