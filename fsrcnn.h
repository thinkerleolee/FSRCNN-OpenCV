#pragma once

#include <array>

#include <tensorconv_ops.h>
#include "fsrcnn_params.h"

//Different model layer countsand filter sizes for FSRCNN vs FSRCNN - s(fast)
//(d, s, m) in paper
//D:Feature_extration Filters
//S:Shrinking-layer Filters
//M:Mapping Layers

class FSRCNN
{
public:
	FSRCNN() {};
	~FSRCNN() {};

	virtual tensorconv::Tensor4D SrOp(tensorconv::Tensor4D y_channel) = 0;
};

class FSRCNN_FAST:public FSRCNN
{
private:
	int scale;

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
	FSRCNN_FAST(int scale);
	~FSRCNN_FAST();

	tensorconv::Tensor4D SrOp(tensorconv::Tensor4D y_channel);
};

class FSRCNN_NORMAL :public  FSRCNN {
private:
	int scale;

	tensorconv::Tensor4DMap feature_extraction_block_feature_extraction_w_;
	tensorconv::Tensor1DMap feature_extraction_block_feature_extraction_b_;

	tensorconv::Tensor1DMap shrinking_block_alpha1_;
	tensorconv::Tensor4DMap shrinking_block_shrinking_w_;
	tensorconv::Tensor1DMap shrinking_block_shrinking_b_;

	tensorconv::Tensor4DMap mapping_block_w3_;
	tensorconv::Tensor1DMap mapping_block_b3_;
	tensorconv::Tensor4DMap mapping_block_w4_;
	tensorconv::Tensor1DMap mapping_block_b4_;
	tensorconv::Tensor1DMap mapping_block_alpha4_;
	tensorconv::Tensor4DMap mapping_block_w5_;
	tensorconv::Tensor1DMap mapping_block_b5_;
	tensorconv::Tensor1DMap mapping_block_alpha5_;
	tensorconv::Tensor4DMap mapping_block_w6_;
	tensorconv::Tensor1DMap mapping_block_b6_;
	tensorconv::Tensor1DMap mapping_block_alpha6_;
	tensorconv::Tensor1DMap mapping_block_alpha7_;
	tensorconv::Tensor4DMap mapping_block_w7_;
	tensorconv::Tensor1DMap mapping_block_b7_;
	tensorconv::Tensor1DMap alpha2_;

	tensorconv::Tensor4DMap expanding_block_w8_;
	tensorconv::Tensor1DMap expanding_block_b8_;
	tensorconv::Tensor1DMap expanding_block_alpha8_;

	tensorconv::Tensor4DMap deconvolution_block_deconv_w_;
	tensorconv::Tensor1DMap deconvolution_block_deconv_b_;
public:
	FSRCNN_NORMAL(int scale);
	~FSRCNN_NORMAL();

	tensorconv::Tensor4D SrOp(tensorconv::Tensor4D y_channel);
};
