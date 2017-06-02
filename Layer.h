/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:42
*/
#ifndef FN_LAYER_H_
#define FN_LAYER_H_

#include <vector>
#include <armadillo>
#include "Blob.h"

namespace fn {

template<typename Dtype>
class Layer
{
public:
	explicit Layer();
	~Layer();

	// Compute the layer output.
	//virtual void Forward(const std::vector<Blob<Dtype>*> &bottom,
	//	const std::vector<Blob<Dtype>*> &top) ;
	//virtual void Forward(const std::vector<arma::Cube<Dtype>*> &bottom,
	//	std::vector<arma::Cube<Dtype>*> &top) =0;
protected:
	std::vector<Layer<Dtype>*> next_;

};

}// namespace fn

#endif // !FN_LAYER_H_