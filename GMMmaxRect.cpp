// GMMmaxRect.cpp : Defines the entry point for the console application.
//������ģ���ҵ��˶�����ľ��ο�
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
	Mat frame;//�洢��Ƶ֡	
	Mat foreground;//�洢ǰ��
	Mat fgdrect;//����ǰ�������ڼ������
	Mat background;//�洢����

	vector<vector<Point> > contours;//����洢�߽�����ĵ�	
	vector<Vec4i> hierarchy;//����洢��ε�����

	//Rect trackWindow;//������ٵľ���
	//RotatedRect trackBox;//����һ����ת�ľ����������CamShift����
	//int hsize = 16;//ÿһάֱ��ͼ�Ĵ�С
	//float hranges[] = { 0, 180 };//hranges�ں���ļ���ֱ��ͼ������Ҫ�õ�
	//int vmin = 10, vmax = 256, smin = 30;
	//const float* phranges = hranges;//

	//Mat hsv, hue, mask, hist, backproj;//image,gray, 

	VideoCapture capture;
	capture.open(0);
	//capture.open("MV.flv");

	if (!capture.isOpened())
	{
		cout << "������ͷʧ��!" << endl;
		return -1;
	}

	CascadeClassifier frontalface_cascade;
	//���ļ���װ��ѵ���õ�Haar����������
	if (!frontalface_cascade.load(cascade_name))//�ж�Haar���������Ƿ�ɹ�
	{
		printf("�޷����ؼ����������ļ���\n");
		return -1;
	}

	BackgroundSubtractorMOG2 mog;//����һ����ϸ�˹��	

	long frameNo = 0;//�洢��Ƶ֡��

	//vector<vector<Rect>> Allfaces;//�洢���е�����
	//vector<Rect> faces;//�洢һ����Ȥ�����ڼ�⵽��������������������Ҫ������whileѭ�����棬�����ſ���ʹ��CamShift����

	while (capture.read(frame))//ѭ����ȡ��һ֡
	{
		++frameNo;//֡����1
		cout << "��" << frameNo << "֡" << endl;//�����ǰ��֡��	

		mog(frame, foreground, 0.01);// �˶�ǰ����⣬�����±�����0.01��ѧϰ���ʣ��ı��С��ı䱳�������ٶ�

		//ȥ������
		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//���ͣ����һ������Ϊ�����������ı��С��Ӱ��ȥ��Ч��
		erode(foreground, foreground, Mat(), Point(-1, -1), 2);//��ʴ
		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//����

		foreground.copyTo(fgdrect);//����ǰ�������ڼ������
		fgdrect = fgdrect > 50;//�������ش���50�����ػ���Ϊ255����������Ϊ0
		//���ǰ��������
		//findContours(fgdrect, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//ֻ��������������
		//findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);//���������������Ϊ����
		findContours(fgdrect, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);//��������������ع����

		vector<Rect> rects;//�洢������������Ӿ��Σ���boundingRect��������
		vector<Mat> frameROIs;//�洢ROI���򣬸������ڷ�����������Ӿ��εĸ���
		int idx = 0;//��������ѭ��
		if (contours.size())//������ϴ��жϣ�������Ƶ��ֻ�б���ʱ�����
		{
			for (; idx >= 0; idx = hierarchy[idx][0])//�ҵ��������������hierarchy[idx][0]��ָ����һ����������û����һ��������hierarchy[idx][0]Ϊ������
			{
				if (fabs(contourArea(Mat(contours[idx]))) > 5000)//�����ǰ������������ڴ�ǰ���������ֵ���򱣴浱ǰֵ
				{
					rects.push_back(boundingRect(contours[idx]));//ѹջ���������������Ӿ���
				}
			}
		}

		for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); it++)//�������з�����������Ӿ���
		{
			rectangle(foreground, *it, Scalar(255, 0, 255), 3, 8, 0);//��ǰ������������������Ӿ��ο�
			rectangle(frame, *it, Scalar(255, 255, 0), 3, 8, 0);//����Ƶ�л���������������Ӿ��ο�
			frameROIs.push_back(Mat(frame, *it));//�洢��Ȥ����
		}
		
		mog.getBackgroundImage(background);   // ���ص�ǰ����ͼ��
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



////opencv2��˹������ģ������opencv1��������ǰ�����ο�
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
//		cout << "������ͷʧ�ܣ�" << endl;
//		return -1;
//	}
//
//	BackgroundSubtractorMOG2 mog;	//cv::BackgroundSubtractorMOG mog;
//
//	Mat frame;
//	Mat foreground;
//	Mat fgdrect;
//	Mat background;
//	Rect rect;//������Ӿ��Σ���boundingRect��������
//
//	long frameNo = 0;
//
//	while (capture.read(frame))
//	{
//		++frameNo;
//		cout << frameNo << endl;//�����ǰ��֡��
//		IplImage frame1 = IplImage(frame);
//		// �˶�ǰ����⣬�����±���
//		mog(frame, foreground, 0.08);
//		//ȥ������
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//���ͣ�3*3��element����������Ϊniters
//		erode(foreground, foreground, Mat(), Point(-1, -1), 2);//��ʴ
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
//		cvFindContours(img_temp, mem_storage, &first_contour, sizeof(CvContour), CV_RETR_EXTERNAL);          //#1 ���������    
//		cvZero(img_temp);
//		cvDrawContours(img_temp, first_contour, cvScalar(100), cvScalar(100), 1);                       //#2 ���������    
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
//		mog.getBackgroundImage(background);   // ���ص�ǰ����ͼ��
//		
//		cvShowImage("ǰ��������ȡ", &frame1);
//		cvShowImage("ǰ��", &img);
//		imshow("����", background);
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



////  ���ڻ�ϸ�˹ģ�͵��˶�Ŀ����
////
//// BackgroundSubtractorMOG ����̨Ӧ�ó���
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
////��һ��gmm,�õ���KaewTraKulPong, P. and R. Bowden (2001).
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
//	dilate(mask, temp, Mat(), Point(-1, -1), niters);//���ͣ�3*3��element����������Ϊniters
//	erode(temp, temp, Mat(), Point(-1, -1), niters * 2);//��ʴ
//	dilate(temp, temp, Mat(), Point(-1, -1), niters);
//
//	findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);//������
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
//	for (; idx >= 0; idx = hierarchy[idx][0])//���û��ô����
//	{
//		const vector<Point>& c = contours[idx];
//		double area = fabs(contourArea(Mat(c)));
//		if (area > maxArea)
//		{
//			maxArea = area;
//			largestComp = idx;//�ҳ����������������
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
//		//bg_model(img, fgmask, update_bg_model ? 0.005 : 0);//����ǰ��maskͼ���������fgmaskΪ8-bit������ͼ�񣬵�3������Ϊѧϰ����
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




//// gmm2_wavetrees.cpp : �������̨Ӧ�ó������ڵ㡣
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
////��һ��gmm,�õ���KaewTraKulPong, P. and R. Bowden (2001).
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
//		double area = fabs(contourArea(Mat(c)));//contourArea�����������
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
//		//bg_model(img, fgmask, update_bg_model ? 0.005 : 0);//����ǰ��maskͼ���������fgmaskΪ8-bit������ͼ�񣬵�3������Ϊѧϰ����
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



