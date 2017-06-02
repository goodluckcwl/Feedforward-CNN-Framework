/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:54:48
*/
#ifndef FN_UTILS_H_
#define FN_UTILS_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <armadillo>

namespace fn {

void clipbboxes(const cv::Mat image, const cv::Rect src, cv::Rect &dst);

// Rotate the image according to two eye centers.
// Obtain the face rectangle according to the eye centers and mouth bottom.
void obtainFaceImage(const cv::Mat src, std::vector<cv::Point2d> landmarks, cv::Rect boundingbox,
	float faceSizeInPupilDistance, cv::Mat & face);

// List files in current directory
void listDir(std::string path, std::vector<std::string> &files);

void saveMat(const char *fileName, cv::Mat mat);

template <typename Dtype>
bool printMat(arma::Mat<Dtype>src, std::string filename);


template <typename Dtype>
bool printCube(arma::Cube<Dtype>src, std::string filename);

//BGR
template <typename Dtype>
void cvMat2Cube(const cv::Mat src, arma::Cube<Dtype>& dst);

template <typename Dtype>
void cube2cvMat(const arma::Cube<Dtype> src, cv::Mat&dst);

template <typename Dtype>
/**
 * [im2col description]
 * @Author   Weiliang                 Chen
 * @DateTime 2016-09-29T10:36:22+0800
 * @param    feats                    [description]
 * @param    kernel_h                 [description]
 * @param    kernel_w                 [description]
 * @param    pad_h                    [description]
 * @param    pad_w                    [description]
 * @param    stride_h                 [description]
 * @param    stride_w                 [description]
 * @param    feat_mat                 [description]
 */
void im2col(const arma::Cube<Dtype> &feats, 
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	arma::Mat<Dtype> &feat_mat);

template <typename Dtype>
/**
 * [col2im description]
 * @Author   Weiliang                 Chen
 * @DateTime 2016-09-29T10:35:45+0800
 * @param    feat_mat                 [description]
 * @param    height                   [description]
 * @param    width                    [description]
 * @param    feats                    [description]
 */
void col2im(const arma::Mat<Dtype> &feat_mat,
	const int height,const int width,
	arma::Cube<Dtype> &feats);

template <typename Dtype>
/**
 * [filter2col ]
 * @Author   Weiliang                 Chen
 * @DateTime 2016-09-29T10:33:57+0800
 * @param    filters                  [description]
 * @param    dst                      [description]
 */
void filter2col(const std::vector<arma::Cube<Dtype>> &filters, arma::Mat<Dtype> &dst);

template <typename Dtype>
bool is_a_ge_zero_and_a_lt_b(Dtype a, Dtype b);
}

#endif // !FN_UTILS_H_

