#include <string>

#include <iostream>
#include <Eigen/Dense>
#include <array>
#include <ctime>
#include "fsrcnn.h"
#include "utils.h"

#include <sys/timeb.h>

#if defined(WIN32)
#include <io.h>
#include <direct.h>
# define  TIMEB    _timeb
# define  ftime    _ftime
#else
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#define TIMEB timeb
#endif

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	/*
		VideoCapture capture;
	capture.open("C:\\Users\\think\\Documents\\个人文档\\Design\\FSRCNN-OpenCV\\video\\the-grandmaster_360p_10s.mp4");//VideoCapture类的方法


	//检查是否打开成功
	if (!capture.isOpened())
	{
		cout << "视频没有打开" << endl;
		return 1;
	}


	//获取视频的帧速率（一般是30或者60）
	double frame_rate = capture.get(CV_CAP_PROP_FPS);
	cout << frame_rate << endl;


	double codec = capture.get(CV_CAP_PROP_FOURCC);

	//获取视频的总帧数目
	long num_frame = static_cast<long>(capture.get(CV_CAP_PROP_FRAME_COUNT));
	cout << num_frame << endl;

	Mat frame;

	//根据帧速率计算帧之间的等待时间，单位ms
	int delat = 1000 / frame_rate;


	int w = 1280;
	int h = 720;

	cv::VideoWriter out(
		"C:\\Users\\think\\Documents\\个人文档\\Design\\FSRCNN-OpenCV\\video\\out-1.mp4", // 输出文件名
		int(codec),  // 编码形式，使用 CV_FOURCC()宏
		int(frame_rate), // 输出视频帧率
		cv::Size(w, h), // 单帧图片的大小
		true // 如果是false,可传入灰度图像 
	);

	int i = 0;
	//循环遍历视频中的全部帧
	while (1)
	{
		capture.read(frame);

		if (!frame.empty())//如果读完就结束
		{
			out.write(fsutils::SR(frame));
			cout << "Finish Frame:" << i++ << endl;
		}
		else
		{
			break;
		}
	}

	capture.release();//不是必须的（由于在VideoCapture类的析构函数中已经调用了）。用于关闭视频文件
	*/
	if (argc != 2) {
		std::cout << "Paramaters Error" << std::endl;
		return 0;
	}
	String path = String("C:\\Users\\think\\Documents\\个人文档\\Design\\FSRCNN-OpenCV\\x64\\Release\\lena10p.bmp");
	Mat img = imread(argv[1]);
	if (img.empty())
	{
		cout << "fail to load image !" << endl;
		return -1;
	}

	int scale = 3;

	FSRCNN_NORMAL sr(scale);

	time_t ltime1, ltime2, tmp_time;
	struct TIMEB tstruct1, tstruct2;

	_ftime64_s(&tstruct1);            // start time ms
	time(&ltime1);               // start time s

	Mat res = fsutils::SR(img, sr, scale);

	time(&ltime2);               // end time sec
	_ftime64_s(&tstruct2);            // end time ms

	tmp_time = (ltime2 * 1000 + tstruct2.millitm) - (ltime1 * 1000 + tstruct1.millitm);
	cout << path << endl;
	cout << "处理时间: " << tmp_time << "ms" << endl;
	if (_access("result", 0) == -1) {
#if defined(WIN32)
		_mkdir("result");
#else
		mkdir("result", 0755);
#endif
		std::cout << "Mkdir:" << "result" << std::endl;
	}

	imwrite("result\\res_" + fsutils::GetPathOrURLShortName(path), res);
	std::cout << "Save as " << ".\result\\res_" + fsutils::GetPathOrURLShortName(path) << std::endl;
	int w = img.cols;
	int h = img.rows;

	Mat bicubic;
	resize(img, bicubic, { w * scale,h * scale }, 0, 0, cv::INTER_CUBIC);

	imwrite("result\\res_bicubic_" + fsutils::GetPathOrURLShortName(path), bicubic);
	std::cout << "Save as " << ".\result\\res_bicubic_" + fsutils::GetPathOrURLShortName(path) << std::endl;


	waitKey(0);
}