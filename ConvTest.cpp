#include <string>

#include <iostream>
#include <Eigen/Dense>
#include <array>
#include <ctime>
#include "fsrcnn.h"
#include "utils.h"

#include <sys/timeb.h>
#if defined(WIN32)
# define  TIMEB    _timeb
# define  ftime    _ftime
#else
#define TIMEB timeb
#endif

using namespace cv;
using namespace std;

int main()
{
	String path = "C:\\Users\\think\\Documents\\个人文档\\lena-img\\lena180p.bmp";
	Mat img = imread(path);
	if (img.empty())
	{
		cout << "fail to load image !" << endl;
		return -1;
	}
	Mat y_img;
	fsutils::PreprocessImg(img, y_img);
	FSRCNN sr;
	tensorconv::Tensor4D im =  fsutils::FromMat2Tenser4D(y_img);

	time_t ltime1, ltime2, tmp_time;
	struct TIMEB tstruct1, tstruct2;

	_ftime64_s(&tstruct1);            // start time ms
	time(&ltime1);               // start time s

	tensorconv::Tensor4D im2 = sr.SrOp(im);

	time(&ltime2);               // end time sec
	_ftime64_s(&tstruct2);            // end time ms

	tmp_time = (ltime2 * 1000 + tstruct2.millitm) - (ltime1 * 1000 + tstruct1.millitm);
	cout << path << endl;
	cout << "处理时间: " << tmp_time << "ms" <<endl;

	Mat f_img = fsutils::FromTensor4D2Mat(im2) * 255;
	f_img.convertTo(f_img, CV_8U);
	imshow("testY", f_img);

	Mat img2 = imread(path);
	cv::Mat img_y_cb_cr;
	cvtColor(img2, img_y_cb_cr, COLOR_BGR2YCrCb);

	std::vector<cv::Mat> img_y_cb_cr_channels(3);
	split(img_y_cb_cr, img_y_cb_cr_channels);
	int w = img2.cols;
	int h = img2.rows;

	resize(img_y_cb_cr_channels[0], img_y_cb_cr_channels[0], { w * 2,h * 2 }, 0, 0, INTER_CUBIC);
	imshow("CUBIC-Y", img_y_cb_cr_channels[0]);

	resize(img_y_cb_cr_channels[1], img_y_cb_cr_channels[1], { w * 2,h * 2 }, 0, 0, INTER_CUBIC);
	resize(img_y_cb_cr_channels[2], img_y_cb_cr_channels[2], { w * 2,h * 2 }, 0, 0, INTER_CUBIC);

	vector<Mat> mv = { img_y_cb_cr_channels[0], img_y_cb_cr_channels[1], img_y_cb_cr_channels[2] };
	cv::Mat dst;
	merge(mv, dst);

	cvtColor(dst, dst, COLOR_YCrCb2BGR);
	imshow("CUBIC", dst);
	vector<Mat> mv1 = { f_img, img_y_cb_cr_channels[1], img_y_cb_cr_channels[2] };
	cv::Mat dst2;
	merge(mv1, dst2);
	cvtColor(dst2, dst2, COLOR_YCrCb2BGR);

	imshow("test", dst2);
	waitKey(0);
}