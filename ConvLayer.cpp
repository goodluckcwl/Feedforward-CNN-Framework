/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#include <armadillo>
#include <iostream>
#include "ConvLayer.h"
#include "Blob.h"
#include "common.h"
#include "Layer.h"
#include "Utils.h"

using std::cout;
using std::endl;

namespace fn {


template<typename Dtype>
ConvLayer<Dtype>::ConvLayer()
{
}

template<typename Dtype>
void ConvLayer<Dtype>::LayerSetUp(const Blob<Dtype> & weights, const arma::Col<Dtype> &bias,
	const int stride_h, const int stride_w,
	const int pad_h, const int pad_w)
{
	std::vector<int> shape = weights.shape();
	kernel_h_ = shape[0];
	kernel_w_ = shape[1];
	channels_ = shape[2];
	number_ = shape[3];
	weights_ = weights;
	bias_ = bias;
	stride_h_ = stride_h;
	stride_w_ = stride_w;
	pad_h_ = pad_h;
	pad_w_ = pad_w;
}

template<typename Dtype>
void ConvLayer<Dtype>:: Forward(const std::vector<arma::Cube<Dtype>*> &bottom,
	std::vector<arma::Cube<Dtype>*> &top) {
	if (bottom.size() != 1 ||top.size() != 1 ) {
		cout << "Error:The dimension of the input data or output data is wrong." << endl;
		return;
	}
	if (bottom[0]->n_rows > 0 
		&& bottom[0]->n_cols > 0 
		&& bottom[0]->n_slices > 0) {
		arma::Mat<Dtype> feat_mat;
		arma::Mat<Dtype> weights_mat;
		bottom[0]->save("input.txt", arma::arma_ascii);
		im2col(*(bottom[0]), kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, feat_mat);
		feat_mat.save("feat_mat.txt",arma::arma_ascii);
		filter2col(*(weights_.data_vec()), weights_mat);
		weights_mat.save("weights_mat.txt", arma::arma_ascii);
		//conv
		arma::Mat<Dtype> conv_mat = feat_mat*weights_mat;
		const int output_h = (bottom[0]->n_rows + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
		const int output_w = (bottom[0]->n_cols + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
		conv_mat.save("conv1_mat.txt",arma::arma_ascii);
		col2im(conv_mat, output_h, output_w, *(top[0]));
		
		//Add bias
		for (int channel = 0; channel < top[0]->n_slices; ++channel) {
			top[0]->slice(channel) += bias_.at(channel);

		}
		
	}
}

template<typename Dtype>
void ConvLayer<Dtype>::Forward(const arma::Cube<Dtype>& bottom, arma::Cube<Dtype>& top)
{
	if (bottom.n_rows > 0
		&& bottom.n_cols > 0
		&& bottom.n_slices > 0) {
		arma::Mat<Dtype> feat_mat;
		arma::Mat<Dtype> weights_mat;
		im2col(bottom, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, feat_mat);
		filter2col(*(weights_.data_vec()), weights_mat);
//		printMat(feat_mat, "feat_mat.txt");
//		printMat(weights_mat, "weights_mat.txt");
		//conv
		arma::Mat<Dtype> conv_mat = feat_mat*weights_mat;
		const int output_h = (bottom.n_rows + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
		const int output_w = (bottom.n_cols + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
		col2im(conv_mat, output_h, output_w, top);

		//Add bias
		for (int channel = 0; channel < top.n_slices; ++channel) {
			top.slice(channel) += bias_.at(channel);
		}

	}
}

template<typename Dtype>
void ConvLayer<Dtype>::CalShape(const arma::Cube<Dtype>& bottom, 
	 std::vector<int>& shape)
{
	const int output_h = (bottom.n_rows + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
	const int output_w = (bottom.n_cols + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
	shape[0] = output_h;
	shape[1] = output_w;
	shape[2] = number_;
}


template<typename Dtype>
ConvLayer<Dtype>::~ConvLayer() {

}

//Explicit instantiation
INSTANTIATE_CLASS(ConvLayer);

}