#pragma once

#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "tensorconv_ops.h"

#include "fsrcnn_s_params.h"

//Different model layer countsand filter sizes for FSRCNN vs FSRCNN - s(fast)
//(d, s, m) in paper
//D:Feature_extration Filters
//S:Shrinking-layer Filters
//M:Mapping Layers
const std::array<int, 3> fsrcnn_model_params{ 56,12,4 }                                                                                                                                                                                                                                               ;
const std::array<int, 3> fsrcnn_s_model_params{ 32,5,1};

class FSRCNN
{
private:
	tensorconv::Tensor4DMap feature_extraction_block_feature_extraction_w_;
	tensorconv::Tensor1DMap feature_extraction_block_feature_extraction_b_;

	tensorconv::Tensor1DMap shrinking_block_alpha1_;
	tensorconv::Tensor4DMap shrinking_block_shrinking_w_;
	tensorconv::Tensor1DMap shrinking_block_shrinking_b_;

	tensorconv::Tensor4DMap mapping_block_w1_;
	tensorconv::Tensor1DMap mapping_block_b1_;
	tensorconv::Tensor1DMap mapping_block_alpha1_;
	tensorconv::Tensor4DMap mapping_block_w2_;
	tensorconv::Tensor1DMap mapping_block_b2_;
	tensorconv::Tensor1DMap mapping_block_alpha2_;

	tensorconv::Tensor4DMap expanding_block_w5_;
	tensorconv::Tensor1DMap expanding_block_b5_;
	tensorconv::Tensor1DMap expanding_block_alpha5_;

	tensorconv::Tensor4DMap deconvolution_block_deconv_w_;
	tensorconv::Tensor1DMap deconvolution_block_deconv_b_;


public:
	FSRCNN();
	~FSRCNN();

	void LoadModel(bool fast = true);
	tensorconv::Tensor4D SrOp(tensorconv::Tensor4D y_channel);
};