/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:51
*/
#ifndef FN_NET_H_
#define FN_NET_H_
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <armadillo>
#include "ConvLayer.h"
#include "ReluLayer.h"
#include "PoolingLayer.h"
#include "InnerProductLayer.h"

namespace fn {

template <typename Dtype>
class Net
{
public:
	Net(std::string name);
	~Net();
	bool init(std::string model_dir);

	void forward(const cv::Mat &image, Dtype *feat, bool flip);
	void forward(const cv::Mat &image, std::vector<Dtype> &feat, bool flip);
	bool load(std::string filename, const int height,
		const int width,const int channels,const int number,
		Blob<Dtype> &weights);
	bool load(std::string filename, const int height,
		const int width, const int channels,
		arma::Cube<Dtype> &mean);
	bool load(std::string filename, const int length, arma::Col<Dtype> &bias);
	bool load(std::string filename, const int height,
		const int width, arma::Mat<Dtype> &weights);
private:
	void forward(const cv::Mat &image, Dtype *feat);

private:
    std::string name_;
	const int IMAGE_SIZE_WIDTH = 47;
	const int IMAGE_SIZE_HEIGHT = 55;
	const int IMAGE_CHANNELS = 1;
	arma::Cube<Dtype> mean_;
	ConvLayer<Dtype> conv1_;
	ConvLayer<Dtype> conv1_1_;
	ConvLayer<Dtype> conv2_1_;
	ConvLayer<Dtype> conv2_2_;
	ConvLayer<Dtype> conv2_;
	ConvLayer<Dtype> conv3_1_;
	ConvLayer<Dtype> conv3_2_;
	ConvLayer<Dtype> conv3_3_;
	ConvLayer<Dtype> conv3_4_;
	ConvLayer<Dtype> conv3_;
	ConvLayer<Dtype> conv4_;

	ReluLayer<Dtype> relu1_;
	ReluLayer<Dtype> relu1_1_;
	ReluLayer<Dtype> relu2_1_;
	ReluLayer<Dtype> relu2_2_;
	ReluLayer<Dtype> relu2_;
	ReluLayer<Dtype> relu3_1_;
	ReluLayer<Dtype> relu3_2_;
	ReluLayer<Dtype> relu3_3_;
	ReluLayer<Dtype> relu3_4_;
	ReluLayer<Dtype> relu3_;
	ReluLayer<Dtype> relu4_;
	
	PoolingLayer<Dtype> pool1_;
	PoolingLayer<Dtype> pool2_;
	PoolingLayer<Dtype> pool3_;

	InnerProductLayer<Dtype> fc160_1_;
	InnerProductLayer<Dtype> fc160_2_;
};


}	//namespace fn
#endif // !FN_NET_H_
