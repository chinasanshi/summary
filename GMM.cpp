// GMM.cpp : Defines the entry point for the console application.
//  ���ڻ�ϸ�˹ģ�͵��˶�Ŀ����

//ʹ��BackgroundSubtractorMOG2

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
	VideoCapture capture;//����һ��VideoCapture��Ķ���
	capture.open(0);// ������ͷ��Ҳ���Ըĳ���Ƶ·����ȡ��Ƶ�ļ��罫0��Ϊ" D:/1.avi"

	if (!capture.isOpened())
	{
		cout << "������ͷʧ�ܣ�" << endl;
		return -1;
	}

	BackgroundSubtractorMOG2 mog;//����һ��BackgroundSubtractorMOG2��Ķ���	

	Mat frame;//������Ƶÿһ֡������
	Mat foreground;//����ǰ��֡����
	Mat background;//���汳��֡����

	long frameNo = 0;//������Ƶ��֡��

	while (capture.read(frame))
	{
		++frameNo;
		cout << frameNo << endl;//�����ǰ��֡��

		// �˶�ǰ����⣬�����±���
		mog(frame, foreground, 0.05);//0.05Ϊ�������ʣ����Լ�����
		//ȥ������
		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//����
		erode(foreground, foreground, Mat(), Point(-1, -1), 2);//��ʴ
		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);

		mog.getBackgroundImage(background);   // ���ص�ǰ����ͼ��

		imshow("video", frame);//��ʾ����ͷ�������Ƶ
		imshow("foreground", foreground);//��ʾǰ��
		imshow("background", background);//��ʾǰ��

		if (waitKey(25) > 0)//����֡��
		{
			break;
		}
	}

	return 0;
}


////ʹ��BackgroundSubtractorMOG
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
//	VideoCapture capture;//����һ��VideoCapture��Ķ���
//	capture.open(0);// ������ͷ��Ҳ���Ըĳ���Ƶ·����ȡ��Ƶ�ļ��罫0��Ϊ" D:/1.avi"
//
//	if (!capture.isOpened())
//	{
//		cout << "������ͷʧ�ܣ�" << endl;
//		return -1;
//	}
//
//	BackgroundSubtractorMOG mog;//����һ��BackgroundSubtractorMOG��Ķ���	
//
//	Mat frame;//������Ƶÿһ֡������
//	Mat foreground;//����ǰ��֡����
//
//	long frameNo = 0;//������Ƶ��֡��
//
//	while (capture.read(frame))
//	{
//		++frameNo;
//		cout << frameNo << endl;//�����ǰ��֡��
//
//		// �˶�ǰ����⣬�����±���
//		mog(frame, foreground, 0.05);//0.05Ϊ�������ʣ����Լ�����
//		//ȥ������
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//����
//		erode(foreground, foreground, Mat(), Point(-1, -1), 2);//��ʴ
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);
//
//		imshow("video", frame);//��ʾ����ͷ�������Ƶ
//		imshow("foreground", foreground);//��ʾǰ��
//
//		if (waitKey(25) > 0)//����֡��
//		{
//			break;
//		}
//	}
//
//	return 0;
//}
