/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:54:27
*/
#ifndef FN_BLOB_H_
#define FN_BLOB_H_
 

#include <string>
#include <vector>
#include <armadillo>
#include "common.h"

namespace fn{

template <typename Dtype>
class Blob
{
public:
	Blob() {};
	explicit Blob(const std::vector<int>& shape);
	explicit Blob(const int height, const int width, const int channels, const int num);

	/**
	 * [Change the dimensions of the blob ]
	 * @Author   Weiliang                 Chen
	 * @DateTime 2016-09-25T11:41:50+0800
	 * @param    shape                    [description]
	 */
	void reshape(const std::vector<int>& shape);

	inline std::string shape_string() const {
        std::ostringstream stream;
        for (int i = 0; i < shape_.size(); ++i) {
        stream << shape_[i] << " ";
        }
        stream << "(" << count_ << ")";
        return stream.str();
	}
	inline const std::vector<int>& shape() const { return shape_; }
	inline int num_axes() const { return shape_.size(); }

	inline  std::vector<arma::Cube<Dtype>> * data_vec() { return &data_; }

	~Blob();
private:
	std::vector<arma::Cube<Dtype>> data_;
	// width,height,channels,number
	std::vector<int> shape_;
	int count_;

	//DISABLE_COPY_AND_ASSIGN(Blob);

}; // class Blob

}	// namespace fn

#endif // !FN_BLOB_H_
