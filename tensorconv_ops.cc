#include "tensorconv_ops.h"

//NHWC
namespace tensorconv {

	Tensor4D Conv2D(const Tensor4D& input,
		const Tensor4D& filters, //[filter_height, filter_width, in_channels, out_channels]
		const PaddingMode padding_mode, const std::array<int, 4>& strides,const Eigen::ThreadPoolDevice& device)
	{
		const int filters_shape[] = { filters.dimension(0),filters.dimension(1),filters.dimension(2),filters.dimension(3) };
		int input_shape[] = { input.dimension(0),input.dimension(1),input.dimension(2),input.dimension(3) };

		int output_shape[] = { input_shape[0],0,0,filters_shape[3] };

		Eigen::PaddingType pad_t = Eigen::PADDING_SAME;
		if (padding_mode == VALID) {
			pad_t = Eigen::PADDING_VALID;
			output_shape[1] = Eigen::divup(float(input.dimension(1) - filters_shape[0] + 1), float(strides[1]));
			output_shape[2] = Eigen::divup(float(input.dimension(2) - filters_shape[1] + 1), float(strides[2]));
		}
		else if (padding_mode == SAME) {
			output_shape[1] = Eigen::divup(float(input.dimension(1)), float(strides[1]));
			output_shape[2] = Eigen::divup(float(input.dimension(2)), float(strides[2]));
		}
		Tensor5D patches(input_shape[0], output_shape[1] * output_shape[2], filters_shape[0], filters_shape[1], input_shape[3]);
		patches.device(device) = input.extract_image_patches(filters_shape[0], filters_shape[1],//patch_rows, patch_cols
			strides[1], strides[2], //row_stride, col_stride
			1, 1, //in_row_stride, in_col_stride
			pad_t);//padding_type
		register int batch = patches.dimension(0);
		register int patche = patches.dimension(1);
		register int patche_h = patches.dimension(2);
		register int patche_w = patches.dimension(3);
		register int channel = patches.dimension(4);

		Tensor4D output = Tensor4D(output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
		output.setZero();
		register int f_b;
		register int b;

		const int CPU_NUM = 8;
		std::mutex output_lock;
		std::array<std::thread*, CPU_NUM> threads;
		int thread_id;
		for (f_b = 0; f_b < filters_shape[3]; ++f_b) { //遍历所有的filter
			for (b = 0; b < batch; ++b) {	//遍历所有的输入
				int gap = patche / CPU_NUM;
				if (patche > CPU_NUM) {
					for (thread_id = 0; thread_id < CPU_NUM; ++thread_id) {
						register int begin = thread_id * gap;
						register int end = (thread_id + 1) * gap;
						if (thread_id == (CPU_NUM - 1)) {
							end = patche;
						}
						threads[thread_id] = new std::thread([begin, end, &patche_h , &patche_w , &channel , &f_b , &b, &output , &output_shape, &patches, &filters]() {
							register int p;
							register int h;
							register int w;
							register int c;
							for (p = begin; p < end; ++p) {	//遍历生成的patch（这里patch过多，造成性能问题！！！）
								//确定这个patch所对应的行列索引,这里是Row-Major有效
								int output_row_index = p / output_shape[2];
								int output_col_index = p % output_shape[2];
								for (h = 0; h < patche_h; ++h) {
									for (w = 0; w < patche_w; ++w) {
										for (c = 0; c < channel; ++c) {
											output(b, output_row_index, output_col_index, f_b)
													+= patches(b, p, h, w, c) * filters(h, w, c, f_b);
										}
									}
								}
							}
							});
					}
					for (thread_id = 0; thread_id < CPU_NUM; ++thread_id) {
						threads[thread_id]->join();
					}
					for (thread_id = 0; thread_id < CPU_NUM; ++thread_id) {
						delete threads[thread_id];
					}
				}
				else {
					register int p;
					register int h;
					register int w;
					register int c;
					for (p = 0; p < patche; ++p) {	//遍历生成的patch（这里patch过多，造成性能问题！！！）
						//确定这个patch所对应的行列索引,这里是Row-Major有效
						int output_row_index = p / output_shape[2];
						int output_col_index = p % output_shape[2];
						for (h = 0; h < patche_h; ++h) {
							for (w = 0; w < patche_w; ++w) {
								for (c = 0; c < channel; ++c) {
									output(b, output_row_index, output_col_index, f_b)
										+= patches(b, p, h, w, c) * filters(h, w, c, f_b);
								}
							}
						}
					}

				}
			}
		}
		return output;
	}

	Tensor4D Depth2Space(const Tensor4D & input, const int blocksize)
	{
		if (blocksize == 0 || input.dimension(3) % (blocksize * blocksize) != 0) {
			std::cerr << "blocksize error" << std::endl;
			abort();
		}
		const int input_shape[] = { input.dimension(0),input.dimension(1),input.dimension(2),input.dimension(3) };

		const int output_shape[] = { input.dimension(0),input.dimension(1) * blocksize,input.dimension(2) * blocksize,input.dimension(3) / (blocksize * blocksize) };

		const int dim_6_shape[] = { output_shape[0],input_shape[1],input_shape[2],blocksize,blocksize,output_shape[3] };
		tensorconv::Tensor6D::Dimensions dim_6_out(dim_6_shape[0], dim_6_shape[1], dim_6_shape[2], dim_6_shape[3], dim_6_shape[4], dim_6_shape[5]);
		std::array<int, 6> shuffling = { 0, 1, 3, 2, 4, 5 };
		tensorconv::Tensor6D resa = input.reshape(dim_6_out).shuffle(shuffling);
		Tensor4D::Dimensions dim(output_shape[0], output_shape[1], output_shape[2], output_shape[3]);
		return resa.reshape(dim);
	}

	Tensor4D Relu(const Tensor4D & src)
	{
		return src.cwiseMax(0.0f);
	}

	Tensor4D PRelu(const Tensor4D & input, const Tensor1D & alphas)
	{
		int input_shape[] = { input.dimension(0),input.dimension(1),input.dimension(2),input.dimension(3) };
		if (input_shape[3] != alphas.dimension(0)) {
			std::cerr << "alpha size error" << std::endl;
			abort();
		}
		Tensor4D temp = Relu(-input);
		for (int b_i = 0; b_i < input_shape[0]; ++b_i) {
			for (int h_i = 0; h_i < input_shape[1]; ++h_i) {
				for (int w_i = 0; w_i < input_shape[2]; ++w_i) {
					for (int c_i = 0; c_i < input_shape[3]; ++c_i) {
						temp(b_i, h_i, w_i, c_i) = temp(b_i, h_i, w_i, c_i) * alphas(c_i);
					}
				}
			}
		}
		return Relu(input) - temp;
	}

	Tensor4D BiasAdd(const Tensor4D & input, const Tensor1D & bias)
	{
		int input_shape[] = { input.dimension(0),input.dimension(1),input.dimension(2),input.dimension(3) };
		if (input_shape[3] != bias.dimension(0)) {
			std::cerr << "bias size error" << std::endl;
			abort();
		}
		Tensor4D temp = input;
		for (int b_i = 0; b_i < input_shape[0]; ++b_i) {
			for (int h_i = 0; h_i < input_shape[1]; ++h_i) {
				for (int w_i = 0; w_i < input_shape[2]; ++w_i) {
					for (int c_i = 0; c_i < input_shape[3]; ++c_i) {
						temp(b_i, h_i, w_i, c_i) = temp(b_i, h_i, w_i, c_i) + bias(c_i);
					}
				}
			}
		}
		return temp;
	}
}
