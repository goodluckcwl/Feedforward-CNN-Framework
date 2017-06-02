/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#include "Layer.h"


namespace fn {

template<typename Dtype>
Layer<Dtype>::Layer() {

}

template<typename Dtype>
Layer<Dtype>::~Layer()
{
}

//Explicit instantiation
INSTANTIATE_CLASS(Layer);

}