// GMMmaxRect.cpp : Defines the entry point for the console application.
//背景建模，找到运动区域的矩形框
#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/background_segm.hpp>
#include "cv.h"
#include <iostream>
#include <string>
#include <vector>
#include "stdio.h"
using namespace std;
using namespace cv;

char* cascade_name = //"haarcascade_frontalface_alt.xml";
//"haarcascade_frontalface_alt_tree.xml";
"C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
//"CascadeClassifier frontalface_cascade";

int main()
{
	Mat frame;//存储视频帧	
	Mat foreground;//存储前景
	Mat fgdrect;//复制前景，用于检测轮廓
	Mat background;//存储背景

	vector<vector<Point> > contours;//定义存储边界所需的点	
	vector<Vec4i> hierarchy;//定义存储层次的向量

	//Rect trackWindow;//定义跟踪的矩形
	//RotatedRect trackBox;//定义一个旋转的矩阵类对象，由CamShift返回
	//int hsize = 16;//每一维直方图的大小
	//float hranges[] = { 0, 180 };//hranges在后面的计算直方图函数中要用到
	//int vmin = 10, vmax = 256, smin = 30;
	//const float* phranges = hranges;//

	//Mat hsv, hue, mask, hist, backproj;//image,gray, 

	VideoCapture capture;
	capture.open(0);
	//capture.open("MV.flv");

	if (!capture.isOpened())
	{
		cout << "打开摄像头失败!" << endl;
		return -1;
	}

	CascadeClassifier frontalface_cascade;
	//从文件中装载训练好的Haar级联分类器
	if (!frontalface_cascade.load(cascade_name))//判断Haar特征加载是否成功
	{
		printf("无法加载级联分类器文件！\n");
		return -1;
	}

	BackgroundSubtractorMOG2 mog;//定义一个混合高斯类	

	long frameNo = 0;//存储视频帧数

	//vector<vector<Rect>> Allfaces;//存储所有的人脸
	//vector<Rect> faces;//存储一个兴趣区域内检测到的人脸。这两个参数需要定义在while循环外面，这样才可以使用CamShift跟踪

	while (capture.read(frame))//循环读取下一帧
	{
		++frameNo;//帧数加1
		cout << "第" << frameNo << "帧" << endl;//输出当前的帧数	

		mog(frame, foreground, 0.01);// 运动前景检测，并更新背景；0.01是学习速率，改变大小会改变背景更新速度

		//去除噪声
		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//膨胀，最后一个参数为迭代次数，改变大小会影响去噪效果
		erode(foreground, foreground, Mat(), Point(-1, -1), 2);//腐蚀
		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//膨胀

		foreground.copyTo(fgdrect);//复制前景，用于检测轮廓
		fgdrect = fgdrect > 50;//所有像素大于50的像素会设为255，其它的设为0
		//检测前景的轮廓
		//findContours(fgdrect, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//只检测最外面的轮廓
		//findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);//检测所有轮廓并分为两层
		findContours(fgdrect, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);//检测所有轮廓并重构层次

		vector<Rect> rects;//存储符合条件的外接矩形，由boundingRect函数返回
		vector<Mat> frameROIs;//存储ROI区域，个数等于符合条件的外接矩形的个数
		int idx = 0;//轮廓个数循环
		if (contours.size())//必须加上此判断，否则当视频中只有背景时会出错
		{
			for (; idx >= 0; idx = hierarchy[idx][0])//找到面积最大的轮廓（hierarchy[idx][0]会指向下一个轮廓，若没有下一个轮廓则hierarchy[idx][0]为负数）
			{
				if (fabs(contourArea(Mat(contours[idx]))) > 5000)//如果当前轮廓的面积大于从前遍历的最大值，则保存当前值
				{
					rects.push_back(boundingRect(contours[idx]));//压栈保存符合条件的外接矩形
				}
			}
		}

		for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); it++)//遍历所有符合条件的外接矩形
		{
			rectangle(foreground, *it, Scalar(255, 0, 255), 3, 8, 0);//在前景画出符合条件的外接矩形框
			rectangle(frame, *it, Scalar(255, 255, 0), 3, 8, 0);//在视频中画出符合条件的外接矩形框
			frameROIs.push_back(Mat(frame, *it));//存储兴趣区域
		}
		
		mog.getBackgroundImage(background);   // 返回当前背景图像
		imshow("video", frame);
		imshow("background", background);
		imshow("foreground", foreground);

		if (waitKey(10) > 0)
		{
			break;
		}
	}

	return 0;
}



////opencv2高斯背景建模，并用opencv1画出最大的前景矩形框
//#include "stdafx.h"
//
//#include "cv.h"    
//#include "highgui.h"  
//#include <opencv2\core\core.hpp>//#include <opencv2\imgproc\imgproc.hpp>
//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/video/background_segm.hpp>
//#include <iostream>
//#include <string>
//#include <vector>
//#include "stdio.h"
//using namespace std;
//using namespace cv;
//int main()
//{
//	VideoCapture capture;
//	capture.open(0);
//
//	if (!capture.isOpened())
//	{
//		cout << "打开摄像头失败！" << endl;
//		return -1;
//	}
//
//	BackgroundSubtractorMOG2 mog;	//cv::BackgroundSubtractorMOG mog;
//
//	Mat frame;
//	Mat foreground;
//	Mat fgdrect;
//	Mat background;
//	Rect rect;//定义外接矩形，由boundingRect函数返回
//
//	long frameNo = 0;
//
//	while (capture.read(frame))
//	{
//		++frameNo;
//		cout << frameNo << endl;//输出当前的帧数
//		IplImage frame1 = IplImage(frame);
//		// 运动前景检测，并更新背景
//		mog(frame, foreground, 0.08);
//		//去除噪声
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//膨胀，3*3的element，迭代次数为niters
//		erode(foreground, foreground, Mat(), Point(-1, -1), 2);//腐蚀
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);
//
//		foreground.copyTo(fgdrect);
//		IplImage img = IplImage(fgdrect);
//		IplImage* img_temp = cvCreateImage(cvGetSize(&img), 8, 1);
//
//		CvMemStorage* mem_storage = cvCreateMemStorage(0);
//		CvSeq *first_contour = NULL, *c = NULL;
//		//vector<vector<CvPoint>>  first_contour; 
//
//		cvThreshold(&img, &img, 128, 255, CV_THRESH_BINARY);
//		img_temp = cvCloneImage(&img);
//		cvFindContours(img_temp, mem_storage, &first_contour, sizeof(CvContour), CV_RETR_EXTERNAL);          //#1 需更改区域    
//		cvZero(img_temp);
//		cvDrawContours(img_temp, first_contour, cvScalar(100), cvScalar(100), 1);                       //#2 需更改区域    
//		c = first_contour;
//		double area = 0;
//		for (; first_contour != 0; first_contour = first_contour->h_next)
//		{
//			if (fabs(cvContourArea(first_contour)) > area)
//			{
//				c = first_contour;
//				area = fabs(cvContourArea(first_contour));
//			}
//		}
//		if (c != NULL)
//		{
//			rect = cvBoundingRect(c, 1);
//			cvRectangle(&frame1, cvPoint(rect.x, rect.y),
//				cvPoint((rect.x + rect.width), (rect.y + rect.height)),
//				CV_RGB(0, 255, 255), 2, 8, 0);
//		}
//		mog.getBackgroundImage(background);   // 返回当前背景图像
//		
//		cvShowImage("前景轮廓提取", &frame1);
//		cvShowImage("前景", &img);
//		imshow("背景", background);
//		
//		if (waitKey(25) > 0)
//		{
//			break;
//		}
//		//cvReleaseImage(&img);
//		cvReleaseImage(&img_temp);
//		cvReleaseMemStorage(&mem_storage);
//	}
//
//	cvDestroyAllWindows();
//	return 0;
//}



////  基于混合高斯模型的运动目标检测
////
//// BackgroundSubtractorMOG 控制台应用程序
////
//
//#include "stdafx.h"
//
//#include "opencv2/core/core.hpp"
//#include "opencv2/video/background_segm.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
////this is a sample for foreground detection functions
//
//Mat img, fgmask, fgimg;
//
////bool update_bg_model = true;
////bool pause = false;
//
////第一种gmm,用的是KaewTraKulPong, P. and R. Bowden (2001).
////An improved adaptive background mixture model for real-time tracking with shadow detection.
//BackgroundSubtractorMOG bg_model;
//
//void refineSegments(const Mat& img, Mat& mask, Mat& dst)
//{
//	int niters = 3;
//
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//
//	Mat temp;
//
//	dilate(mask, temp, Mat(), Point(-1, -1), niters);//膨胀，3*3的element，迭代次数为niters
//	erode(temp, temp, Mat(), Point(-1, -1), niters * 2);//腐蚀
//	dilate(temp, temp, Mat(), Point(-1, -1), niters);
//
//	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);//找轮廓
//	
//	dst = Mat::zeros(img.size(), CV_8UC3);
//
//	if (contours.size() == 0)
//		return;
//
//	// iterate through all the top-level contours,
//	// draw each connected component with its own random color
//	int idx = 0, largestComp = 0;
//	double maxArea = 0;
//
//	for (; idx >= 0; idx = hierarchy[idx][0])//这句没怎么看懂
//	{
//		const vector<Point>& c = contours[idx];
//		double area = fabs(contourArea(Mat(c)));
//		if (area > maxArea)
//		{
//			maxArea = area;
//			largestComp = idx;//找出包含面积最大的轮廓
//		}
//	}
//	Scalar color(0, 255, 0);
//	drawContours(dst, contours, largestComp, color, CV_FILLED, 8, hierarchy);
//}
//
//int main()
//{
//	VideoCapture capture;
//	capture.open(0);
//	
//	if (!capture.isOpened())
//	{
//		std::cout << "read video failure" << std::endl;
//		return -1;
//	}
//	//frame.copyTo(img);
//	for (;;)
//	{
//		capture >> img;
//		if (img.empty())
//		{
//			break;
//		}
//
//		//update the model
//		//bg_model(img, fgmask, update_bg_model ? 0.005 : 0);//计算前景mask图像，其中输出fgmask为8-bit二进制图像，第3个参数为学习速率
//		bg_model(img, fgmask, 0.005);
//		refineSegments(img, fgmask, fgimg);
//
//		imshow("image", img);
//		imshow("foreground image", fgimg);		
//
//		if (waitKey(25) > 0)
//		{
//			break;
//		}
//	}
//
//	return 0;
//}




//// gmm2_wavetrees.cpp : 定义控制台应用程序的入口点。
////
//
//#include "stdafx.h"
//
//#include "opencv2/core/core.hpp"
//#include "opencv2/video/background_segm.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
////this is a sample for foreground detection functions
//
//Mat img, fgmask, fgimg, background;
//
//bool update_bg_model = true;
//bool pause = false;
//
////第一种gmm,用的是KaewTraKulPong, P. and R. Bowden (2001).
////An improved adaptive background mixture model for real-time tracking with shadow detection.
//BackgroundSubtractorMOG2 bg_model;
//
//void refineSegments(const Mat& img, Mat& mask, Mat& dst)
//{
//	int niters = 3;
//
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//
//	Mat temp;
//
//	dilate(mask, temp, Mat(), Point(-1, -1), niters);
//	erode(temp, temp, Mat(), Point(-1, -1), niters * 2);
//	dilate(temp, temp, Mat(), Point(-1, -1), niters);
//
//	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
//
//	dst = Mat::zeros(img.size(), CV_8UC3);
//
//	if (contours.size() == 0)
//		return;
//
//	// iterate through all the top-level contours,
//	// draw each connected component with its own random color
//	int idx = 0, largestComp = 0;
//	double maxArea = 0;
//
//	for (; idx >= 0; idx = hierarchy[idx][0])
//	{
//		//const vector<Point>& c = contours[idx];
//		const vector<Point>& c = contours[idx];
//		double area = fabs(contourArea(Mat(c)));//contourArea计算区域面积
//		if (area > maxArea)
//		{
//			maxArea = area;
//			largestComp = idx;
//		}
//	}
//	Scalar color(255, 0, 0);
//	drawContours(dst, contours, largestComp, color, CV_FILLED, 8, hierarchy);
//}
//
//int main()
//{
//	VideoCapture capture;
//	Mat frame;
//	capture.open(0);
//	int i=0;
//
//	if (!capture.isOpened())
//	{
//		std::cout << "read video failure" << std::endl;
//		return -1;
//	}
//	for (;;)
//	{
//		capture >> frame;
//		frame.copyTo(img);
//		i++;
//		if (i == 100)
//		{
//			i = 0;
//		}
//		//update the model
//		//bg_model(img, fgmask, update_bg_model ? 0.005 : 0);//计算前景mask图像，其中输出fgmask为8-bit二进制图像，第3个参数为学习速率
//		bg_model(img, fgmask, 0.005);
//		if (i % 10 == 0)
//		{
//			refineSegments(img, fgmask, fgimg);
//		}
//		//refineSegments(img, fgmask, fgimg);
//
//		bg_model.getBackgroundImage(background);
//
//		imshow("foreground image", fgimg);
//		imshow("background", background);
//		imshow("image", img);
//
//		char k = (char)waitKey(25);
//		if (k == 27) break;
//
//		if (k == ' ')
//		{
//			pause = !pause;
//		}
//	}
//
//	return 0;
//}



