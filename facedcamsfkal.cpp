// facedcamsfkal.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <ctype.h>
#include <stdio.h>

using namespace std;
using namespace cv;

//static CascadeClassifier frontalface_cascade = 0;

char* cascade_name =
"C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
/* "haarcascade_profileface.xml";*/

void help()
{
	cout << "\n���ǻ���Haar������������⣬��ʹ��CamShift���и��ٵ�һ��demo\n"
		"������Զ�������������и��٣����Լ�Ⲣ���ٶ������\n"
		"����������ͷ�ɼ���Ƶ����ʼ����Ƶ��ʱ�����һ����������û�м�⵽����Ҫ���¼�ⰴ����d������\n";

	cout << "\n\n���ܼ�: \n"
		"\tEsc - �˳�����\n"
		"\tc - ֹͣ����\n"
		"\td - ���¼������\n";
}

int main()
{
	help();

	CascadeClassifier frontalface_cascade;
	//���ļ���װ��ѵ���õ�Haar����������
	if (!frontalface_cascade.load(cascade_name))//�ж�Haar���������Ƿ�ɹ�
	{
		printf("�޷����ؼ����������ļ���\n");
		return -1;
	}	

	int camwidth=640, camheight = 480;//��������ͷͼ��Ŀ�Ⱥ͸߶�
	int detction = 1;//�����Ƿ�ʹ��������⺯���������
	int trackObject = 0; //�����Ƿ��ڸ���Ŀ��
	Rect trackWindow;//������ٵľ���
	RotatedRect trackBox;//����һ����ת�ľ��������
	int hsize = 16;//ÿһάֱ��ͼ�Ĵ�С
	float hranges[] = { 0, 180 };//hranges�ں���ļ���ֱ��ͼ������Ҫ�õ�
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//
	Mat frame, image, gray, hsv, hue, mask, hist, backproj;// histimg = Mat::zeros(200, 320, CV_8UC3), 
	//Mat image, gray;//����ѭ�������������ν

	VideoCapture capture;
	capture.open(0);//Ҳ����д��VideoCapture capture(0);
	if (!capture.isOpened())
	{
		printf("������ͷʧ�ܣ�\n");
		return -1;
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, camwidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, camheight);//����ͼ����Ⱥ͸߶�

	namedWindow("�������Ƶ");
	vector<Rect> faces;//ֻ�ܷ���whileѭ�����棬��������
	int facenum;
	while (1)
	{
		capture >> frame;//��ȡ��Ƶ����һ֡
		if (frame.empty())
		{
			break;
		}
		frame.copyTo(image); 

		if (detction)//��ʼ��Ϊ1�����Ի��Ƚ���һ���������
		{
			detction = 0;//Haar�����������detctionһֱ����Ϊ0����˸�if����ֻ��ִ��һ�Σ��������°�����ĸd
			trackObject = 1;
			cvtColor(image, gray, CV_BGR2GRAY);//��rgb����ͷ֡ת���ɻҶ�ͼ
			equalizeHist(gray, gray);

			double t = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
			//�����������������������ο�
			frontalface_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));			
			t = (double)cvGetTickCount() - t;//��ü�����������ʱ�䲢���
			printf("���������ʱ��Ϊ = %gms/n\n", t / ((double)cvGetTickFrequency()*1000.));
			facenum = faces.size();
		}

		if (trackObject && (!faces.empty()))//���trackObject=1�Ҽ�⵽���������if����
		{
			for (int i = 0; i < facenum; i++)
			{
				cvtColor(image, hsv, CV_BGR2HSV);//��rgb����ͷ֡ת����hsv�ռ��
				//inRange�����Ĺ����Ǽ����������ÿ��Ԫ�ش�С�Ƿ���2��������ֵ֮�䣬�����ж�ͨ��,mask����0ͨ������Сֵ��Ҳ����h����
				//����������hsv��3��ͨ�����Ƚ�h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)�����3��ͨ�����ڶ�Ӧ�ķ�Χ�ڣ���
				//mask��Ӧ���Ǹ����ֵȫΪ1(0xff)������Ϊ0(0x00).
				inRange(hsv, Scalar(0, smin, 10), Scalar(180, 256, 256), mask);
				int ch[] = { 0, 0 };
				hue.create(hsv.size(), hsv.depth());//hue��ʼ��Ϊ��hsv��С���һ���ľ���ɫ���Ķ������ýǶȱ�ʾ�ģ�������֮�����120�ȣ���ɫ���180��
				mixChannels(&hsv, 1, &hue, 1, ch, 1);//��hsv��һ��ͨ��(Ҳ����ɫ��)�������Ƶ�hue�У�0��������

				//�˴��Ĺ��캯��roi�õ���Mat hue�ľ���ͷ����roi������ָ��ָ��hue����������ͬ�����ݣ�selectΪ�����Ȥ������
				//Mat roi(hue, selection), maskroi(mask, selection);//mask�����hsv����Сֵ
				trackWindow = faces.at(i);
				Mat roi(hue, trackWindow), maskroi(mask, trackWindow);//mask�����hsv����Сֵ

				//calcHist()������һ������Ϊ����������У���2��������ʾ����ľ�����Ŀ����3��������ʾ��������ֱ��ͼά��ͨ�����б���4��������ʾ��ѡ�����뺯��
				//��5��������ʾ���ֱ��ͼ����6��������ʾֱ��ͼ��ά������7������Ϊÿһάֱ��ͼ����Ĵ�С����8������Ϊÿһάֱ��ͼbin�ı߽�
				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);//��roi��0ͨ������ֱ��ͼ��ͨ��mask����hist�У�hsizeΪÿһάֱ��ͼ�Ĵ�С
				normalize(hist, hist, 0, 255, CV_MINMAX);//��hist����������鷶Χ��һ��������һ����0~255

				calcBackProject(&hue, 1, 0, hist, backproj, &phranges);//����ֱ��ͼ�ķ���ͶӰ������hueͼ��0ͨ��ֱ��ͼhist�ķ���ͶӰ��������backproj��
				backproj &= mask;

				//opencv2.0�Ժ�İ汾��������ǰû��cv�����ˣ������������������2����˼�ĵ���Ƭ����ɵĻ�����ǰ���Ǹ�Ƭ�β����ɵ��ʣ����һ����ĸҪ
				//��д������Camshift�������һ����ĸ�Ǹ����ʣ���Сд������meanShift�����ǵڶ�����ĸһ��Ҫ��д
				trackBox = CamShift(backproj, trackWindow,               //trackWindowΪ���ѡ�������TermCriteriaΪȷ��������ֹ��׼��
					TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));//CV_TERMCRIT_EPS��ͨ��forest_accuracy,CV_TERMCRIT_ITER

				if (trackWindow.area() <= 1)                                                  //��ͨ��max_num_of_trees_in_the_forest  
				{
					int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
					trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
						trackWindow.x + r, trackWindow.y + r) &
						Rect(0, 0, cols, rows);//Rect����Ϊ�����ƫ�ƺʹ�С������һ��������Ϊ��������Ͻǵ����꣬�����ĸ�����Ϊ����Ŀ�͸�
				}

				ellipse(image, trackBox, Scalar(255, 255, 0), 3, CV_AA);//���ٵ�ʱ������ԲΪ����Ŀ��				
			}
		}

		imshow("�������Ƶ", image);

		char c = (char)waitKey(10);

		switch (c)
		{
			case 'c':            //����trackObject���Ӷ����ٸ���Ŀ�����
				trackObject = 0;
				break;
			case 'd':       //ʹ��Haar�����������
				detction = 1;
				break;
			default:   
				break;
		}

		if (c == 27)
		{
			break;
		}

	}

	return 0;
}


////�õ� kalman �˲�����Ԥ�⣬���������ܵ�����Χ���� camshift ����
////�ȳ�ʼ��
//const int stateDim = 4;
//const int measureDim = 2;
//const int contralDim = 0;
//KalmanFilter KF;
//KF.init(stateDim, measureDim, contralDim, CV_32F);
//Mat state(stateDim, 1, CV_32F); //Դ����û�����
//Mat processNoise(stateDim, 1, CV_32F);
//Mat measurement = Mat::zeros(measureDim, 1, CV_32F);
//CvRNG rng = cvRNG(-1);
//float A[stateDim][stateDim] = //transition matrix
//{
//	1, 0, 25, 0,
//	0, 1, 0, 25,
//	0, 0, 1, 0,
//	0, 0, 0, 1
//};
//randn(state, Scalar::all(0), Scalar::all(0.1));
//KF.transitionMatrix = *(Mat_<float>(2, 2) << 1, 1, 0, 1);
//setIdentity(KF.measurementMatrix);
//setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
//setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
//setIdentity(KF.errorCovPost, Scalar::all(1));
//randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
////initialize post state of kalman filter at random




//cvtColor(image, image, CV_BGR2GRAY);
//int ROIx, ROIy, ROIw, ROIh;//ROI�����Ͻǵ�����λ��(ROIx,ROIy)�����ROIw���߶�ROIy
//ROIx = 51;
//ROIy = 62;
//ROIw = 634;
//ROIh = 100;
//Rect ROIRect = { ROIx, ROIy, ROIw, ROIh };//����ROI,����ƥ���ʱ��
//Mat imageROI = small_image(ROIRect);

//Mat imageROI_gray;
//cvtColor(imageROI, imageROI_gray, CV_BGR2GRAY);
//equalizeHist(imageROI_gray, imageROI_gray);//ͼ��ֱ��ͼ���⻯����ǿͼ��Աȶ�