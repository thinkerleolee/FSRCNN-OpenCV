#include "fsrcnn.h"



FSRCNN::FSRCNN():
	feature_extraction_block_feature_extraction_w_(fsrcnn_s_params::getInstance()->feature_extraction_block_feature_extraction_w, 5, 5, 1, 32),
	feature_extraction_block_feature_extraction_b_(fsrcnn_s_params::getInstance()->feature_extraction_block_feature_extraction_b, 32),
	shrinking_block_alpha1_(fsrcnn_s_params::getInstance()->shrinking_block_alpha1, 32),
	shrinking_block_shrinking_w_(fsrcnn_s_params::getInstance()->shrinking_block_shrinking_w, 1, 1, 32, 5),
	shrinking_block_shrinking_b_(fsrcnn_s_params::getInstance()->shrinking_block_shrinking_b, 5),
	mapping_block_w1_(fsrcnn_s_params::getInstance()->mapping_block_w3, 3, 3, 5, 5),
	mapping_block_b1_(fsrcnn_s_params::getInstance()->mapping_block_b3, 5),
	mapping_block_alpha1_(fsrcnn_s_params::getInstance()->mapping_block_alpha4, 5),
	mapping_block_w2_(fsrcnn_s_params::getInstance()->mapping_block_w4, 1, 1, 5, 5),
	mapping_block_b2_(fsrcnn_s_params::getInstance()->mapping_block_b4, 5),
	mapping_block_alpha2_(fsrcnn_s_params::getInstance()->alpha2, 5),
	expanding_block_w5_(fsrcnn_s_params::getInstance()->expanding_block_w5, 1, 1, 5, 32),
	expanding_block_b5_(fsrcnn_s_params::getInstance()->expanding_block_b5, 32),
	expanding_block_alpha5_(fsrcnn_s_params::getInstance()->expanding_block_alpha5, 32),
	deconvolution_block_deconv_w_(fsrcnn_s_params::getInstance()->deconvolution_block_deconv_w, 3, 3, 32, 4),
	deconvolution_block_deconv_b_(fsrcnn_s_params::getInstance()->deconvolution_block_deconv_b, 4)
{

}

FSRCNN::~FSRCNN()
{
}

void FSRCNN::LoadModel(bool fast)
{
}

tensorconv::Tensor4D FSRCNN::SrOp(tensorconv::Tensor4D input)
{
	// Create the Eigen ThreadPoolDevice.
	Eigen::ThreadPool* tp = new Eigen::ThreadPool(8);
	Eigen::ThreadPoolDevice my_device(tp, 8);

	tensorconv::Tensor4D output = tensorconv::Conv2D(input, feature_extraction_block_feature_extraction_w_, tensorconv::VALID, { 1,1,1,1 }, my_device);
	output = tensorconv::BiasAdd(output, feature_extraction_block_feature_extraction_b_);

	output = tensorconv::PRelu(output, shrinking_block_alpha1_);
	output = tensorconv::Conv2D(output, shrinking_block_shrinking_w_, tensorconv::SAME, { 1,1,1,1 }, my_device);
	output = tensorconv::BiasAdd(output, shrinking_block_shrinking_b_);

	tensorconv::Tensor4D temp = output;

	output = tensorconv::Conv2D(output, mapping_block_w1_, tensorconv::SAME, { 1,1,1,1 }, my_device);
	output = tensorconv::BiasAdd(output, mapping_block_b1_);
	output = tensorconv::PRelu(output, mapping_block_alpha1_);
	output = tensorconv::Conv2D(output, mapping_block_w2_, tensorconv::SAME, { 1,1,1,1 }, my_device);
	output = tensorconv::BiasAdd(output, mapping_block_b2_);

	output += temp;

	output = tensorconv::PRelu(output, mapping_block_alpha2_);

	output = output = tensorconv::Conv2D(output, expanding_block_w5_, tensorconv::SAME, { 1,1,1,1 }, my_device);
	output = tensorconv::BiasAdd(output, expanding_block_b5_);
	output = tensorconv::PRelu(output, expanding_block_alpha5_);

	output = tensorconv::Conv2D(output, deconvolution_block_deconv_w_, tensorconv::SAME, { 1,1,1,1 }, my_device);

	output = tensorconv::BiasAdd(output, deconvolution_block_deconv_b_);

	output = tensorconv::Depth2Space(output, 2);
	return output;

}

