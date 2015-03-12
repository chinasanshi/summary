// GMM.cpp : Defines the entry point for the console application.
//  基于混合高斯模型的运动目标检测

//使用BackgroundSubtractorMOG2

#include "stdafx.h"
#include <string>
#include "opencv2/video/background_segm.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
using namespace std;
using namespace cv;

int main()
{
	VideoCapture capture;//定义一个VideoCapture类的对象
	capture.open(0);// 打开摄像头，也可以改成视频路径读取视频文件如将0改为" D:/1.avi"

	if (!capture.isOpened())
	{
		cout << "打开摄像头失败！" << endl;
		return -1;
	}

	BackgroundSubtractorMOG2 mog;//定义一个BackgroundSubtractorMOG2类的对象	

	Mat frame;//保存视频每一帧的数据
	Mat foreground;//保存前景帧数据
	Mat background;//保存背景帧数据

	long frameNo = 0;//保存视频的帧数

	while (capture.read(frame))
	{
		++frameNo;
		cout << frameNo << endl;//输出当前的帧数

		// 运动前景检测，并更新背景
		mog(frame, foreground, 0.05);//0.05为更新速率，可自己调整
		//去除噪声
		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//膨胀
		erode(foreground, foreground, Mat(), Point(-1, -1), 2);//腐蚀
		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);

		mog.getBackgroundImage(background);   // 返回当前背景图像

		imshow("video", frame);//显示摄像头拍摄的视频
		imshow("foreground", foreground);//显示前景
		imshow("background", background);//显示前景

		if (waitKey(25) > 0)//控制帧率
		{
			break;
		}
	}

	return 0;
}


////使用BackgroundSubtractorMOG
//
//#include "stdafx.h"
//#include <string>
//#include "opencv2/video/background_segm.hpp"
//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <vector>
//#include <stdio.h>
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	VideoCapture capture;//定义一个VideoCapture类的对象
//	capture.open(0);// 打开摄像头，也可以改成视频路径读取视频文件如将0改为" D:/1.avi"
//
//	if (!capture.isOpened())
//	{
//		cout << "打开摄像头失败！" << endl;
//		return -1;
//	}
//
//	BackgroundSubtractorMOG mog;//定义一个BackgroundSubtractorMOG类的对象	
//
//	Mat frame;//保存视频每一帧的数据
//	Mat foreground;//保存前景帧数据
//
//	long frameNo = 0;//保存视频的帧数
//
//	while (capture.read(frame))
//	{
//		++frameNo;
//		cout << frameNo << endl;//输出当前的帧数
//
//		// 运动前景检测，并更新背景
//		mog(frame, foreground, 0.05);//0.05为更新速率，可自己调整
//		//去除噪声
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//膨胀
//		erode(foreground, foreground, Mat(), Point(-1, -1), 2);//腐蚀
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);
//
//		imshow("video", frame);//显示摄像头拍摄的视频
//		imshow("foreground", foreground);//显示前景
//
//		if (waitKey(25) > 0)//控制帧率
//		{
//			break;
//		}
//	}
//
//	return 0;
//}
