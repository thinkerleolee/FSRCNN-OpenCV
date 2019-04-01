#include "fsrcnn.h"



FSRCNN_FAST::FSRCNN_FAST(int scale = 2):
	scale(scale),
	feature_extraction_block_feature_extraction_w_(fsrcnn_s_params::getInstance()->get_feature_extraction_block_feature_extraction_w(scale), 5, 5, 1, 32),
	feature_extraction_block_feature_extraction_b_(fsrcnn_s_params::getInstance()->get_feature_extraction_block_feature_extraction_b(scale), 32),
	shrinking_block_alpha1_(fsrcnn_s_params::getInstance()->get_shrinking_block_alpha1(scale), 32),
	shrinking_block_shrinking_w_(fsrcnn_s_params::getInstance()->get_shrinking_block_shrinking_w(scale), 1, 1, 32, 5),
	shrinking_block_shrinking_b_(fsrcnn_s_params::getInstance()->get_shrinking_block_shrinking_b(scale), 5),
	mapping_block_w1_(fsrcnn_s_params::getInstance()->get_mapping_block_w3(scale), 3, 3, 5, 5),
	mapping_block_b1_(fsrcnn_s_params::getInstance()->get_mapping_block_b3(scale), 5),
	mapping_block_alpha1_(fsrcnn_s_params::getInstance()->get_mapping_block_alpha4(scale), 5),
	mapping_block_w2_(fsrcnn_s_params::getInstance()->get_mapping_block_w4(scale), 1, 1, 5, 5),
	mapping_block_b2_(fsrcnn_s_params::getInstance()->get_mapping_block_b4(scale), 5),
	mapping_block_alpha2_(fsrcnn_s_params::getInstance()->get_alpha2(scale), 5),
	expanding_block_w5_(fsrcnn_s_params::getInstance()->get_expanding_block_w5(scale), 1, 1, 5, 32),
	expanding_block_b5_(fsrcnn_s_params::getInstance()->get_expanding_block_b5(scale), 32),
	expanding_block_alpha5_(fsrcnn_s_params::getInstance()->get_expanding_block_alpha5(scale), 32),
	deconvolution_block_deconv_w_(fsrcnn_s_params::getInstance()->get_deconvolution_block_deconv_w(scale), 3, 3, 32, scale * scale),
	deconvolution_block_deconv_b_(fsrcnn_s_params::getInstance()->get_deconvolution_block_deconv_b(scale), scale * scale)
{

}

FSRCNN_FAST::~FSRCNN_FAST()
{
}

void FSRCNN_FAST::LoadModel(int scale)
{
}

tensorconv::Tensor4D FSRCNN_FAST::SrOp(tensorconv::Tensor4D input)
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

	output = tensorconv::Depth2Space(output, scale);
	return output;

}

void FSRCNN_NORMAL::LoadModel(int scale)
{
}

tensorconv::Tensor4D FSRCNN_NORMAL::SrOp(tensorconv::Tensor4D y_channel)
{
	return tensorconv::Tensor4D();
}
