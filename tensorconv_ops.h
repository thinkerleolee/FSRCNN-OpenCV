#pragma once

#include <memory>
#include <iostream>
#include <array>

#define EIGEN_USE_THREADS

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/TensorSymmetry>
#include <unsupported/Eigen/CXX11/ThreadPool>


namespace tensorconv {
	//NHWC
	using Tensor6D = Eigen::Tensor<float, 6, Eigen::RowMajor>;
	using Tensor5D = Eigen::Tensor<float, 5, Eigen::RowMajor>;
	using Tensor4D = Eigen::Tensor<float, 4, Eigen::RowMajor>;
	using Tensor3D = Eigen::Tensor<float, 3, Eigen::RowMajor>;
	using Tensor2D = Eigen::Tensor<float, 2, Eigen::RowMajor>;
	using Tensor1D = Eigen::Tensor<float, 1, Eigen::RowMajor>;

	using Tensor4DMap = Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>>;
	using Tensor1DMap = Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>>;


	enum PaddingMode {
		SAME = 0,
		VALID = 1
	};

	namespace ops {
	}

	Tensor4D Conv2D(const Tensor4D& input, const Tensor4D& filters,
						const PaddingMode padding_mode,const std::array<int, 4>& strides, const Eigen::ThreadPoolDevice& device);

	Tensor4D Depth2Space(const Tensor4D& input, const int blocksize);

	Tensor4D Relu(const Tensor4D& src);

	Tensor4D PRelu(const Tensor4D& src, const Tensor1D& alphas);

	Tensor4D BiasAdd(const Tensor4D& src, const Tensor1D& bias);

}
