/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#include <vector>
#include "Blob.h"
#include "common.h"


namespace fn {

//template <typename Dtype>
//Blob<Dtype>::Blob()
//{
//}

template <typename Dtype>
Blob<Dtype>::Blob(const std::vector<int>& shape) {
	NOT_IMPLEMENTED
}

template<typename Dtype>
Blob<Dtype>::Blob(const int height, const int width, const int channels, const int num)
{
	shape_.push_back(height);
	shape_.push_back(width);
	shape_.push_back(channels);
	shape_.push_back(num);
	for (int n = 0; n < num; ++n) {
		arma::Cube<Dtype> data_block(height, width, channels);
		data_.push_back(data_block);
	}
}

template <typename Dtype>
void Blob<Dtype>::reshape(const std::vector<int>& shape) {
	NOT_IMPLEMENTED
	
}

template <typename Dtype>
Blob<Dtype>::~Blob()
{
}

//Explicit instantiation
INSTANTIATE_CLASS(Blob);


}