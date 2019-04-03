#include <string>

#include <iostream>
#include <Eigen/Dense>
#include <array>
#include <numeric>
#include <chrono>
#include <unordered_map>

#include "fsrcnn.h"
#include "utils.h"

#if defined(WIN32)
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

using namespace cv;
using namespace std;

static void show_usage(std::string name)
{
	std::cerr << "Usage: " << name << " <option(s)> SOURCE-IMG" << std::endl
		<< "Options:\n"
		<< "\t-f,--fast True/Flase\tUse FSRCNN-s OR FSRCNN\n"
		<< "\t-s,--scale 2/3\tSpecify the scale num"
		<< std::endl;
}

int main(int argc, char** argv)
{
	if (argc != 6) {
		show_usage(argv[0]);
		return 1;
	}
	std::unordered_map<string, string> paras;

	for (int i = 1; i < argc; ++i) {
		if (i == argc - 1) {
			paras["source"] = String(argv[i]);
		}
		else {
			paras[String(argv[i])] = String(argv[i+1]);
			++i;
		}
	}
	bool fast;
	int scale;
	String path;

	if (paras.count("--fast") || paras.count("-f")) {
		auto func_fast = [&](String str) {
			if ("True" == str || "TRUE" == str || "true" == str) {
				fast = true;
			}
			else if("False" == str || "FALSE" == str || "false" == str) {
				fast = false;
			}
			else {
				show_usage(argv[0]);
				exit(1);
			}
		};
		if (paras.count("--fast")) {
			func_fast(paras["--fast"]);
		}
		else if(paras.count("-f")){
			func_fast(paras["-f"]);
		}
	}
	else {
		show_usage(argv[0]);
		return 1;
	}
	
	if (paras.count("--scale") || paras.count("-s")) {
		auto func_fast = [&](String str) {
			if ("2" == str) {
				scale = 2;
			}
			else if ("3" == str) {
				scale = 3;
			}
			else {
				show_usage(argv[0]);
				exit(1);
			}
		};
		if (paras.count("--scale")) {
			func_fast(paras["--scale"]);
		}
		else if (paras.count("-s")) {
			func_fast(paras["-s"]);
		}
	}
	else {
		show_usage(argv[0]);
		return 1;
	}
	if (paras.count("source")) {
		path = paras["source"];
	}
	else {
		show_usage(argv[0]);
		return 1;
	}

	std::cout << "fast: " << fast << std::endl;
	std::cout << "scale: " << scale << std::endl;
	std::cout << "source: " << path << std::endl;

	//String path = String("C:\\Users\\think\\Documents\\¸öÈËÎÄµµ\\Design\\FSRCNN-OpenCV\\x64\\Release\\lena10p.bmp");

	Mat img = imread(path);
	if (img.empty())
	{
		cout << "fail to load image !" << endl;
		return -1;
	}

	FSRCNN *sr;
	if (fast) {
		sr = new FSRCNN_FAST(scale);
	}
	else {
		sr = new FSRCNN_NORMAL(scale);
	}

	auto t1 = std::chrono::system_clock::now();

	Mat res = fsutils::SR(img, *sr, scale);

	auto t2 = std::chrono::system_clock::now();
	cout << path << endl;
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	cout << "Prosessing Time: " << diff.count() << "ms" << endl;
	if (_access("result", 0) == -1) {
#if defined(WIN32)
		_mkdir("result");
#else
		mkdir("result", 0755);
#endif
		std::cout << "Mkdir:" << "result" << std::endl;
	}

	imwrite("result\\res_fsrcnn_" + fsutils::GetPathOrURLShortName(path), res);
	std::cout << "Save as " << ".\\result\\res_fsrcnn_" + fsutils::GetPathOrURLShortName(path) << std::endl;
	int w = img.cols;
	int h = img.rows;

	Mat bicubic;
	resize(img, bicubic, { w * scale,h * scale }, 0, 0, cv::INTER_CUBIC);

	imwrite("result\\res_bicubic_" + fsutils::GetPathOrURLShortName(path), bicubic);
	std::cout << "Save as " << ".\\result\\res_bicubic_" + fsutils::GetPathOrURLShortName(path) << std::endl;

	std::cout << "Press any key to continue..." << std::endl;
	getchar();
}