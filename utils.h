#pragma once

#include <opencv2/opencv.hpp>
#include "tensorconv_ops.h"
#include <vector>

#define DEBUG

namespace fsutils {
	void PaddingImg(cv::Mat& src, cv::Mat& dest,
		int top, int bottom, int left, int right)
	{
		left = (left > 0 ? left : 0);
		right = (right > 0 ? right : 0);
		top = (top > 0 ? top : 0);
		bottom = (bottom > 0 ? bottom : 0);
		cv::copyMakeBorder(src, dest, top, bottom, left, right, cv::BORDER_REFLECT);
	}

	//Preprocess single image file
	//(1) Get image's Y channel
	//(2) Padding
	//(3) Normalize
	void  PreprocessImg(cv::Mat & src_img, cv::Mat & dest) {
		cv::Mat img_y_cb_cr;
		cvtColor(src_img, img_y_cb_cr, cv::COLOR_BGR2YCrCb);
		std::vector<cv::Mat> img_y_cb_cr_channels(3);
		split(img_y_cb_cr, img_y_cb_cr_channels);
		PaddingImg(img_y_cb_cr_channels[0], dest, 2, 2, 2, 2);
		//transform datatype from uchar to float
		dest.convertTo(dest, CV_32FC1);
		dest = dest / 255.0;
	}

	tensorconv::Tensor4D FromMat2Tenser4D(const cv::Mat& src_img) {
		tensorconv::Tensor4D res(1, src_img.rows, src_img.cols, 1);
		for (int row = 0; row < src_img.rows; ++row) {
			for (int col = 0; col < src_img.cols; ++col) {
				res(0, row, col, 0) = src_img.at<float>(row, col);
			}
		}
		return res;
	}

	cv::Mat FromTensor4D2Mat(const tensorconv::Tensor4D& tensor) {
		cv::Mat res(cv::Size(tensor.dimension(2), tensor.dimension(1)), CV_32FC1, cv::Scalar(0));
		for (int row = 0; row < tensor.dimension(1); ++row) {
			for (int col = 0; col < tensor.dimension(2); ++col) {
				res.at<float>(row, col) = tensor(0, row, col, 0);
			}
		}
		return res;
	}
}
