/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <armadillo>
#include "Net.h"
#include "ConvLayer.h"
#include "Blob.h"
#include "common.h"
#include "Layer.h"
#include "Utils.h"

using std::ifstream;
using std::stringstream;
using std::string;
using std::vector;
using std::cout;
using std::endl;

namespace fn {


template <typename Dtype>
Net<Dtype>::Net(std::string name) {
    name_ = name;
}

template <typename Dtype>
Net<Dtype>::~Net()
{
}

template<typename Dtype>
bool Net<Dtype>::init(std::string model_dir)
{
	//Load mean image.
	mean_ = arma::Cube<Dtype>(IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_CHANNELS);
	if (!load(model_dir+"/mean.txt", IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, IMAGE_CHANNELS, mean_)) {
		return false;
	}

	//conv1
	Blob<Dtype> w1(4,4,1,32);
	arma::Col<Dtype> b1(32);
	if (!load(model_dir+"/w1.txt", 4, 4, 1, 32, w1)) {
		return false;
	}
	if (!load(model_dir+"/b1.txt", 32, b1)) {
		return false;
	}
	conv1_.LayerSetUp(w1, b1, 1, 1, 0, 0);

	//conv1_1
	Blob<Dtype> w1_1(3, 3, 32, 32);
	arma::Col<Dtype> b1_1(32);
	if (!load(model_dir + "/w1_1.txt", 3, 3, 32, 32, w1_1)) {
		return false;
	}
	if (!load(model_dir + "/b1_1.txt", 32, b1_1)) {
		return false;
	}
	conv1_1_.LayerSetUp(w1_1, b1_1, 1, 1, 1, 1);

	//pool1
	pool1_.LayerSetUp(2, 2, 2, 2, 0, 0);

	//conv2_1
	Blob<Dtype> w2_1(3, 3, 32, 32);
	arma::Col<Dtype>b2_1(32);
	if (!load(model_dir + "/w2_1.txt", 3, 3, 32, 32, w2_1)) {
		return false;
	}
	if (!load(model_dir + "/b2_1.txt", 32, b2_1)) {
		return false;
	}
	conv2_1_.LayerSetUp(w2_1, b2_1, 1, 1, 1, 1);

	//conv2_2
	Blob<Dtype> w2_2(3, 3, 32, 32);
	arma::Col<Dtype>b2_2(32);
	if (!load(model_dir + "/w2_2.txt", 3, 3, 32, 32, w2_2)) {
		return false;
	}
	if (!load(model_dir + "/b2_2.txt", 32, b2_2)) {
		return false;
	}
	conv2_2_.LayerSetUp(w2_2, b2_2, 1, 1, 1, 1);

	//conv2
	Blob<Dtype> w2(3,3,32,64);
	arma::Col<Dtype>b2(64);
	if (!load(model_dir+"/w2.txt", 3, 3, 32, 64, w2)) {
		return false;
	}
	if (!load(model_dir+"/b2.txt", 64, b2)) {
		return false;
	}
	conv2_.LayerSetUp(w2, b2, 1, 1, 0, 0);

	//pool2
	pool2_.LayerSetUp(2, 2, 1, 1, 0, 0);

	//conv3_1
	Blob<Dtype> w3_1(3, 3, 64, 64);
	arma::Col<Dtype>b3_1(64);
	if (!load(model_dir + "/w3_1.txt", 3, 3, 64, 64, w3_1)) {
		return false;
	}
	if (!load(model_dir + "/b3_1.txt", 64, b3_1)) {
		return false;
	}
	conv3_1_.LayerSetUp(w3_1, b3_1, 1, 1, 1, 1);

	//conv3_2
	Blob<Dtype> w3_2(3, 3, 64, 64);
	arma::Col<Dtype>b3_2(64);
	if (!load(model_dir + "/w3_2.txt", 3, 3, 64, 64, w3_2)) {
		return false;
	}
	if (!load(model_dir + "/b3_2.txt", 64, b3_2)) {
		return false;
	}
	conv3_2_.LayerSetUp(w3_2, b3_2, 1, 1, 1, 1);

	//conv3_3
	Blob<Dtype> w3_3(3, 3, 64, 64);
	arma::Col<Dtype>b3_3(64);
	if (!load(model_dir + "/w3_3.txt", 3, 3, 64, 64, w3_3)) {
		return false;
	}
	if (!load(model_dir + "/b3_3.txt", 64, b3_3)) {
		return false;
	}
	conv3_3_.LayerSetUp(w3_3, b3_3, 1, 1, 1, 1);

	//conv3_4
	Blob<Dtype> w3_4(3, 3, 64, 64);
	arma::Col<Dtype>b3_4(64);
	if (!load(model_dir + "/w3_4.txt", 3, 3, 64, 64, w3_4)) {
		return false;
	}
	if (!load(model_dir + "/b3_4.txt", 64, b3_4)) {
		return false;
	}
	conv3_4_.LayerSetUp(w3_4, b3_4, 1, 1, 1, 1);

	//conv3
	Blob<Dtype> w3(3, 3, 64, 96);
	arma::Col<Dtype>b3(96);
	if (!load(model_dir+"/w3.txt", 3, 3, 64, 96, w3)) {
		return false;
	}
	if (!load(model_dir+"/b3.txt", 96, b3)) {

	}
	conv3_.LayerSetUp(w3, b3, 1, 1, 0, 0);

	//pool3
	pool3_.LayerSetUp(2, 2, 2, 2, 0, 0);

	//conv4
	Blob<Dtype> w4(2, 2, 96, 128);
	arma::Col<Dtype>b4(128);
	if (!load(model_dir+"/w4.txt", 2, 2, 96, 128, w4)) {
		return false;
	}
	if (!load(model_dir+"/b4.txt", 128, b4)) {
		return false;
	}
	conv4_.LayerSetUp(w4, b4, 1, 1, 0, 0);

	//fc160_1
	arma::Mat<Dtype>fc160_w1(9504,160);
	arma::Col<Dtype>fc160_b1(160);
	if (!load(model_dir+"/fc160_b1.txt", 160, fc160_b1)) {
		return false;
	}
	if (!load(model_dir+"/fc160_w1.txt", 9504, 160, fc160_w1)) {
		return false;
	}
	fc160_1_.LayerSetUp(fc160_w1, fc160_b1);

	//fc160_2
	arma::Mat<Dtype>fc160_w2(10240, 160);
	arma::Col<Dtype>fc160_b2(160);
	if (!load(model_dir+"/fc160_b2.txt", 160, fc160_b2)) {
		return false;
	}
	if (!load(model_dir+"/fc160_w2.txt", 10240, 160, fc160_w2)) {
		return false;
	}
	fc160_2_.LayerSetUp(fc160_w2, fc160_b2);
	return true;
}

template<typename Dtype>
void Net<Dtype>::forward(const cv::Mat & input, Dtype *feat160)
{
	cv::Mat input_gray;
	if (!input.data) {
		cout <<"No image data." << endl;
		return;
	}
	if (abs(input.rows*1.0 / input.cols - IMAGE_SIZE_HEIGHT*1.0 / IMAGE_SIZE_WIDTH) > 0.01) {
		cout << "Warning:The length-width ratio of the input image is not 55/47.Result may be inaccurate." << endl;
	}
	if (input.channels() != 1) {
		cv::cvtColor(input, input_gray, CV_BGR2GRAY);
	}
	else {
		input_gray = input;
	}
	
	/*******************************Data Transform******************************************/
	cv::Mat image;
	cv::resize(input_gray, image, cv::Size(IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT));
	arma::Cube<Dtype> conv1_input = arma::Cube<Dtype>(image.rows, image.cols, image.channels());
	cvMat2Cube(image, conv1_input);
	//Subtract the mean image.

	conv1_input = conv1_input - mean_;
	cube2cvMat(conv1_input, image);
	image.convertTo(image, CV_8UC3);
	cv::Mat im_mean;
	cube2cvMat(mean_, im_mean);
	im_mean.convertTo(im_mean, CV_8UC3);

	/****************************************************************************************/
	//conv1
	vector<int> shape;
	shape.resize(3);
	conv1_.CalShape(conv1_input, shape);
	arma::Cube<Dtype > conv1_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv1_.Forward(conv1_input, conv1_output);
	//relu1
	relu1_.Forward(conv1_output, conv1_output);
//	printCube(conv1_output, "conv1_output.txt");

	//conv1_1
	conv1_1_.CalShape(conv1_output, shape);
	arma::Cube<Dtype> conv1_1_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv1_1_.Forward(conv1_output, conv1_1_output);
	//relu1_1
	relu1_1_.Forward(conv1_1_output, conv1_1_output);
//	printCube(conv1_1_output, "conv1_1_output.txt");
	
	//pool1
	pool1_.CalShape(conv1_1_output, shape);
	arma::Cube<Dtype > pool1_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	pool1_.Forward(conv1_1_output, pool1_output);
//	printCube(pool1_output, "pool1_output.txt");

	//conv2_1
	conv2_1_.CalShape(pool1_output, shape);
	arma::Cube<Dtype> conv2_1_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv2_1_.Forward(pool1_output, conv2_1_output);
//	printCube(conv2_1_output, "conv2_1_output1.txt");
	//relu2_1
	relu2_1_.Forward(conv2_1_output, conv2_1_output);
//	printCube(conv2_1_output, "conv2_1_output.txt");

	//conv2_2
	conv2_2_.CalShape(conv2_1_output, shape);
	arma::Cube<Dtype> conv2_2_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv2_2_.Forward(conv2_1_output, conv2_2_output);
	//relu2_2
	relu2_2_.Forward(conv2_2_output, conv2_2_output);

	//res2_2
	conv2_2_output = conv2_2_output + pool1_output;

	//conv2
	conv2_.CalShape(conv2_2_output, shape);
	arma::Cube<Dtype> conv2_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv2_.Forward(conv2_2_output, conv2_output);
	//relu2
	relu2_.Forward(conv2_output, conv2_output);
//	printCube(conv2_output, "conv2_output.txt");

	//pool2
	pool2_.CalShape(conv2_output, shape);
	arma::Cube<Dtype> pool2_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	pool2_.Forward(conv2_output, pool2_output);
//	printCube(pool2_output, "pool2_output.txt");

	//conv3_1
	conv3_1_.CalShape(pool2_output, shape);
	arma::Cube<Dtype> conv3_1_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv3_1_.Forward(pool2_output, conv3_1_output);
	//relu3_1
	relu3_1_.Forward(conv3_1_output, conv3_1_output);

	//conv3_2
	conv3_2_.CalShape(conv3_1_output, shape);
	arma::Cube<Dtype> conv3_2_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv3_2_.Forward(conv3_1_output, conv3_2_output);
	//relu3_2
	relu3_2_.Forward(conv3_2_output, conv3_2_output);

	//res3_2
	conv3_2_output = conv3_2_output + pool2_output;

	//conv3_3
	conv3_3_.CalShape(conv3_2_output, shape);
	arma::Cube<Dtype> conv3_3_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv3_3_.Forward(conv3_2_output, conv3_3_output);
	//relu3_3
	relu3_3_.Forward(conv3_3_output, conv3_3_output);

	//conv3_4
	conv3_4_.CalShape(conv3_3_output, shape);
	arma::Cube<Dtype> conv3_4_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv3_4_.Forward(conv3_3_output, conv3_4_output);
	//relu3_4
	relu3_4_.Forward(conv3_4_output, conv3_4_output);

	//res3_4
	conv3_4_output = conv3_4_output + conv3_2_output;

	//conv3
	conv3_.CalShape(conv3_4_output, shape);
	arma::Cube<Dtype> conv3_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv3_.Forward(conv3_4_output, conv3_output);
	//relu3
	relu3_.Forward(conv3_output, conv3_output);

	//pool3
	pool3_.CalShape(conv3_output, shape);
	arma::Cube<Dtype> pool3_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	pool3_.Forward(conv3_output, pool3_output);

	//conv4
	conv4_.CalShape(pool3_output, shape);
	arma::Cube<Dtype> conv4_output = arma::Cube<Dtype>(shape[0], shape[1], shape[2]);
	conv4_.Forward(pool3_output, conv4_output);

	//relu4
	relu4_.Forward(conv4_output, conv4_output);

	//fc160_1
	arma::Col<Dtype> fc160_1_output = arma::Col<Dtype>(160);
	fc160_1_.Forward(pool3_output, fc160_1_output);

	//fc160_2
	arma::Col<Dtype> fc160_2_output = arma::Col<Dtype>(160);
	fc160_2_.Forward(conv4_output, fc160_2_output);

	//fc160
	float sum = 0;
	for (int i = 0; i < fc160_1_output.n_elem; ++i) {
		feat160[i] = fc160_1_output.at(i) + fc160_2_output.at(i);
		sum += feat160[i];
	}


}

template<typename Dtype>
void Net<Dtype>::forward(const cv::Mat &image, std::vector<Dtype> &feat, bool flip){
	cv::Mat im_flip;
	int length;
	Dtype *pf = NULL;
	if(flip){
		length = 320;
		cv::flip(image, im_flip, 1);
		pf = (Dtype *)malloc(sizeof(Dtype)*length);
		forward(image, pf);
		forward(im_flip, pf + 160);

	}
	else{
		length = 160;
		pf = (Dtype *)malloc(sizeof(Dtype)*length);
		forward(im_flip, pf);

	}


	//Normalization
	Dtype sum = 0;
	for (int i = 0; i <length; ++i) {
		sum += pf[i]*pf[i];
	}
	sum = sqrt(sum);

	feat.clear();
	for (int i = 0; i < length; ++i) {
		feat.push_back(pf[i] / sum) ;
	}

	free(pf);
}

template<typename Dtype>
void Net<Dtype>::forward(const cv::Mat & image, Dtype * feat, bool flip)
{
	cv::Mat im_flip;
	int length;
	if (flip) {
		cv::flip(image, im_flip, 1);
		forward(image, feat);
		forward(im_flip, feat + 160);
		length = 320;
	}
	else {
		forward(im_flip, feat);
		length = 160;
	}

	//Normalization
	Dtype sum = 0;
	for (int i = 0; i <length; ++i) {
		sum += feat[i]*feat[i];
	}
	sum = sqrt(sum);
	for (int i = 0; i < length; ++i) {
		feat[i] = feat[i] / sum;
	}
}

//
template<typename Dtype>
bool Net<Dtype>::load(std::string filename, 
	const int height, const int width, const int channels, const int number, 
	Blob<Dtype>& weights)
{
	ifstream in(filename);
	if (!in.is_open()) {
		cout << "Cannot open " <<filename<<"."<< endl;
		return false;
	}
	stringstream ss;
	vector<arma::Cube<Dtype>> *filters=weights.data_vec();
	filters->clear();
	for (int n = 0; n < number; ++n) {
		arma::Cube<Dtype> filter = arma::Cube<Dtype>(height, width, channels);
		Dtype *data = filter.memptr();
		const int channel_size = height*width;
		for (int c = channels; c --; data+=channel_size) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					char buffer[256];
					in >> buffer;
					Dtype value = 0;
					value=atof(buffer);
					*(data+x*height+y) = value;
				}
			}
		}
		filters->push_back(filter);
	}
	return true;
}

template<typename Dtype>
bool Net<Dtype>::load(std::string filename, const int height, const int width, const int channels, arma::Cube<Dtype>& mean)
{
	ifstream in(filename);
	if (!in.is_open()) {
		cout << "Cannot open " << filename << "." << endl;
		return false;
	}
	Dtype *data = mean.memptr();
	Dtype value;
	const int channel_size = mean.n_elem_slice;
	for (int c = channels; c--; data += channel_size) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				char buffer[256];
				in >> buffer;
				value = atof(buffer);
				*(data + x*height + y) = value;
			}
		}
		int k = 0;
	}
	return true;
}

template<typename Dtype>
bool Net<Dtype>::load(std::string filename, const int length, arma::Col<Dtype>& bias)
{
	ifstream in(filename);
	if (!in.is_open()) {
		cout << "Cannot open " << filename << "." << endl;
		return false;
	}
	bias = arma::Col<Dtype>(length);
	Dtype *data = bias.memptr();
	for (int i = 0; i < length; ++i) {
		char buffer[256];
		in >> buffer;
		Dtype value;
		value = atof(buffer);
		*(data++) = value;
	}
	return true;
}

template<typename Dtype>
bool Net<Dtype>::load(std::string filename, const int height, const int width, arma::Mat<Dtype>& weights)
{
	ifstream in(filename);
	if (!in.is_open()) {
		cout << "Cannot open " << filename << "." << endl;
		return false;
	}
	Dtype *data = weights.memptr();
	Dtype value;
	for (int w = 0; w < width; ++w) {
		for (int h = 0; h < height; ++h) {
			char buffer[256];
			in >> buffer;
			value = atof(buffer);
			*(data + w*height + h) = value;
		}
	}
	return true;
}


//Explicit instantiation
INSTANTIATE_CLASS(Net);

}	// namespace fn