/*
* @Author: Weiliang Chen
* @Date:   2016-09-29 10:51:07
* @Last Modified by:   Weiliang Chen
* @Last Modified time: 2016-09-29 10:55:27
*/
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include "Utils.h"
#include "common.h"

using std::cout;
using std::endl;
using std::ofstream;
using cv::FileStorage;
using cv::Mat;
using cv::Point2d;
using cv::Rect;

namespace fn {

void clipbboxes(const Mat image, const Rect src, Rect &dst) {
	int x1 = MAX(src.x, 0);
	int y1 = MAX(src.y, 0);
	int x2 = MIN(src.x + src.width - 1, image.cols - 1);
	int y2 = MIN(src.y + src.height - 1, image.rows - 1);
	dst = Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
}

void obtainFaceImage(const cv::Mat src, std::vector<cv::Point2d> landmarks, cv::Rect boundingbox,
	float faceSize, cv::Mat & face)
{
	cv::Scalar blue = cv::Scalar(255, 0, 0);
	cv::Scalar red = cv::Scalar(0, 0, 255);
	cv::Scalar green = cv::Scalar(0, 255, 0);
	Point2d eyeCenter;
	Point2d left = landmarks[0];
	Point2d right = landmarks[1];
	Point2d nose = landmarks[2];
	const double ZOOM_MULTIPLES = 1.0;

	eyeCenter.x = (left.x + right.x)*0.5f;
	eyeCenter.y = (left.y + right.y)*0.5f;

	//Get the angle between the eyes
	double dy = (right.y - left.y);
	double dx = (right.x - left.x);
	double pupilDistance = sqrt(dx*dx + dy*dy);
	double angle = atan2(dy, dx)*180.0 / CV_PI;      //Convert from radians to degrees.

	Mat rot_mat = cv::getRotationMatrix2D(eyeCenter, angle, ZOOM_MULTIPLES);
	Mat dst;
	cv::warpAffine(src, dst, rot_mat, dst.size());

	Point2d newLeft = Point2d(eyeCenter.x - pupilDistance / 2, eyeCenter.y);
	Point2d newRight = Point2d(eyeCenter.x + pupilDistance / 2, eyeCenter.y);
	

	//cv::circle(dst,newLeft,3,cv::Scalar(255,0,0),2);
	//cv::circle(dst,newRight,3,cv::Scalar(255,0,0),2);
	//cv::circle(dst,newNose,3,blue,2);
	//cv::circle(dst,newBottom,3,cv::Scalar(255,0,0),2);
	//cv::circle(dst,left,3,red,2);
	//cv::circle(dst,right,3,red,2);
	//cv::circle(dst,nose,3,red,2);
	//cv::circle(dst,bottom,3,red,2);
	// Calculate the face rectangle
	double width = boundingbox.width;
	double height = boundingbox.height;
	if (height / width > 55.0 / 47.0) {
		width = height * 47 / 55;
	}
	else {
		height = width * 55 / 47;
	}
	width = width*faceSize;
	height = height*faceSize;

	Rect faceRect = Rect(round(eyeCenter.x-width/2), round(eyeCenter.y-height*0.4), 
		round(width), round(height));
	clipbboxes(src, faceRect, faceRect);
	face = dst(faceRect);
	/*cv::rectangle(dst,faceRect,blue);
	cv::imshow("obtaintest", dst);
	cvWaitKey(1);*/
}

void listDir(std::string path, std::vector<std::string> &files){
    DIR *pDir;

    struct dirent *ent;
    int i=0;
    char childpath[512];
    pDir = opendir(path.c_str());
    memset(childpath,0,sizeof(childpath));

    while((ent=readdir(pDir))!=NULL){
        if(ent->d_type&DT_DIR){
            if(strcmp(ent->d_name,".")==0||strcmp(ent->d_name,"..")==0){
                continue;
            }
            sprintf(childpath,"%s",ent->d_name);
            files.push_back(std::string(childpath));
        }
        else{
            sprintf(childpath,"%s",ent->d_name);
            files.push_back(std::string(childpath));
        }
    }
    closedir(pDir);
}

	//��ͨ��Mat����ɶ�ά����
//�豣֤�����ַ����
void saveMat(const char *fileName, Mat mat) {
	FileStorage fs(fileName, FileStorage::WRITE);
	if (!fs.open(fileName, FileStorage::WRITE)) {
		fprintf(stderr, "Error:The model file cannot be open.\n");
		return;
	}
	if (!mat.isContinuous()) {
		fprintf(stderr, "Error:The matrix is not continue.\n");
		return;
	}
	Mat t = Mat(mat.rows, mat.cols*mat.channels(), mat.depth(), mat.data);
	fs << "mat" << t;

	fs.release();
	return;

}


template<typename Dtype>
bool printMat(arma::Mat<Dtype> src, std::string filename)
{
	ofstream out(filename);
	if (!out.is_open()) {
		cout << "Cannot open file " << filename << endl;
		return false;
	}
	int rows = src.n_rows;
	int cols = src.n_cols;
	out << "width:" << cols << " height:" << rows << endl;
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			Dtype data = (Dtype)src.at(x*rows + y);
			out << data << " ";
		}
		out << endl;
	}
	out.close();
	return true;
}
// Explicit instantiation
template bool printMat<float>(arma::Mat<float> src, std::string filename);

template<typename Dtype>
bool printCube(arma::Cube<Dtype> src, std::string filename)
{
	ofstream out(filename);
	if (!out.is_open()) {
		cout <<"Cannot open file "<<filename << endl;
		return false;
	}
	int rows = src.n_rows;
	int cols = src.n_cols;
	int slices = src.n_slices;
	int step = src.n_elem_slice;
	out << "width:" << cols << " height:" << rows << " slices:" << slices << endl;
	for (int c = 0; c < slices; ++c) {
		out << "slices "<<c<<":" << endl;
		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				Dtype data = (Dtype)src.at(c*step + x*rows + y);
				out << data<<" ";
			}
			out << endl;
		}
	}
	out.close();
	return true;
}
// Explicit instantiation
template bool printCube<float>(arma::Cube<float> src, std::string filename);

template<typename Dtype>
void cvMat2Cube(const cv::Mat src, arma::Cube<Dtype>& dst)
{
	int rows = src.rows;
	int cols = src.cols;
	int channels = src.channels();
	cv::Mat tmp;
	if (channels == 3) {
		src.convertTo(tmp, CV_32FC3);
	}
	else if (channels == 1) {
		src.convertTo(tmp, CV_32FC1);
	}
	else {
		cout << "The input image has wrong depth." << endl;

		return;
	}
	dst = arma::Cube<Dtype>(rows, cols, channels);
	float *s1 = (float *)tmp.data;
	int step = rows*cols;
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			for (int c = 0; c < channels; ++c) {
				dst.at(c*step + x*rows + y) = *s1++;
			}
		}
	}
}
// Explicit instantiation
template void cvMat2Cube<float>(const cv::Mat src, arma::Cube<float>& dst);
//template void cvMat2Cube<double>(const cv::Mat src, arma::Cube<double>& dst);

template<typename Dtype>
void cube2cvMat(const arma::Cube<Dtype>src, cv::Mat &dst) {
	int rows = src.n_rows;
	int cols = src.n_cols;
	int slices = src.n_slices;
	//float
	dst = cv::Mat(rows, cols, CV_32FC(slices));
	int step = src.n_elem_slice;
	Dtype *dst_data = (Dtype *)dst.data;
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			for (int c = 0; c < slices; ++c) {
				*dst_data++ = (Dtype)src.at(c*step + x*rows + y);
			}
		}
	}
}
// Explicit instantiation
template void cube2cvMat<float>(const arma::Cube<float>src, cv::Mat &dst);
//template void cube2cvMat<double>(const arma::Cube<double>src, cv::Mat &dst);

template<typename Dtype>
void im2col(const arma::Cube<Dtype> &feats,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	arma::Mat<Dtype>& feat_mat)
{
	const int height = feats.n_rows;
	const int width = feats.n_cols;
	const int channels = feats.n_slices;
	const int output_h = (height + 2 * pad_h - kernel_h )/ stride_h + 1;
	const int output_w = (width + 2 * pad_w - kernel_w )/ stride_w + 1;

	const int channel_size = height*width;
	feat_mat = arma::Mat<Dtype>(output_h*output_w, kernel_w*kernel_h*channels);
	const Dtype *data_im = feats.memptr();
	Dtype *data_col = feat_mat.memptr();
	//����armadillo��ľ���Ԫ�����ڴ����ǰ����ȱ��������Ų�����Ԫ��������������
	//���ǰ�һ�ξ���ĸ�Ԫ��չ����һ�У���һ�������չ����һ�С�
	//�����������֮�󣬵õ��ľ����ÿһ����һ��feature map.
	for (int c = channels; c--; data_im += channel_size) {
		//����������.���մ����ң����ϵ��µ�˳��һ���˵�Ԫ���ų�һ��
		for (int kernel_y = 0; kernel_y < kernel_h; kernel_y++) {
			for (int kernel_x = 0; kernel_x < kernel_w; kernel_x++) {
				//����Feature Map���������ͼ�����ƶ��������ң����ϵ��¡�
				//��ǰ��ͼ���е�λ��
				int input_y = kernel_y-pad_h;
				for (int output_y = output_h; output_y; output_y--) {
					//����Ƿ���paddingλ��
					if (!is_a_ge_zero_and_a_lt_b(input_y, height)) {
						for (int output_x = output_w; output_x; output_x--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_x = kernel_x - pad_w;
						for (int output_x = output_w; output_x; output_x--) {
							if (is_a_ge_zero_and_a_lt_b(input_x, width)) {
								*(data_col++) = data_im[input_x*height + input_y];
							}
							else {
								*(data_col++) = 0;
							}
							input_x += stride_w;
						}
					}
					input_y += stride_h;
				}
			}
		}
	}
}

// Explicit instantiation
template void im2col<float>(const arma::Cube<float> &feats,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	arma::Mat<float>& feat_mat);
template void im2col<double>(const arma::Cube<double> &feats,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	arma::Mat<double>& feat_mat);

template<typename Dtype>
void col2im(const arma::Mat<Dtype> &feat_mat, 
	const int height, const int width,
	arma::Cube<Dtype> &feats) {

	const int channels = feat_mat.n_cols;
	const int channel_size = height*width;
	//feats = arma::Cube<Dtype>(height, width, channels);
	const Dtype *data_col = feat_mat.memptr();
	Dtype *data_dst = feats.memptr();
	for (int c = channels; c--; data_dst+=channel_size) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				*(data_dst + x*height + y) = *data_col;
				data_col++;
			}
		}
	}
}
// Explicit instantiation
template void col2im<float>(const arma::Mat<float> &feat_mat, 
	const int height, const int width,
	arma::Cube<float> &feats);
template void col2im<double>(const arma::Mat<double> &feat_mat,
	const int height, const int width,
	arma::Cube<double> &feats);

template<typename Dtype>
void filter2col(const std::vector<arma::Cube<Dtype>> &filters, arma::Mat<Dtype> &dst)
{
	int numberOfFilters = filters.size();
	int rows = filters[0].n_rows;
	int cols = filters[0].n_cols;
	int channels = filters[0].n_slices;
	int channel_size = filters[0].n_elem_slice;
	dst = arma::Mat<Dtype>(rows*cols*channels, numberOfFilters);
	Dtype *data_col = dst.memptr();
	for (int number = 0; number < numberOfFilters; ++number) {
		Dtype *data_src = (Dtype *)(filters[number].memptr());
		for (int c = channels; c--; data_src += channel_size) {
			for (int kernel_y = 0; kernel_y < rows; ++kernel_y) {
				for (int kernel_x = 0; kernel_x < cols; ++kernel_x) {
					*(data_col++) = *(data_src + kernel_x*rows + kernel_y);
				}
			}
		}
	}
}
// Explicit instantiation
template void filter2col<float>(const std::vector<arma::Cube<float>>& cube_vec, arma::Mat<float>& dst);
template void filter2col<double>(const std::vector<arma::Cube<double>>& cube_vec, arma::Mat<double>& dst);

template<typename Dtype>
bool is_a_ge_zero_and_a_lt_b(Dtype a, Dtype b)
{
	if (0 <= a&&a < b) {
		return true;
	}
	return false;
}
// Explicit instantiation
template bool is_a_ge_zero_and_a_lt_b<int>(int a, int b);
template bool is_a_ge_zero_and_a_lt_b<float>(float a, float b);
template bool is_a_ge_zero_and_a_lt_b<double>(double a, double b);
} //namespace fn