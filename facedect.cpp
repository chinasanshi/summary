// facedect.cpp : Defines the entry point for the console application.
//
//�Լ�ʹ��opencv2��д�Ĵ��� 
#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
//#include <vector>
#include <stdio.h>

using namespace std;
using namespace cv;

//static CascadeClassifier frontalface_cascade = 0;

char* cascade_name =
"C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
/* "haarcascade_profileface.xml";*/

int _tmain()
{
	CascadeClassifier frontalface_cascade;
	//���ļ���װ��ѵ���õ�Haar����������
	if (!frontalface_cascade.load(cascade_name))//�ж�Haar���������Ƿ�ɹ�
	{
		printf("�޷����ؼ����������ļ���\n");
		return -1;
	}
	HOGDescriptor hog;//����HOG����,����Ĭ�ϲ���
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector()); //����SVM�������������Ѿ�ѵ���õ����˼�������
	vector<Rect> regions,regionsRect;

	namedWindow("result", 1);

	Mat image;
	image = imread("face.jpg");//����ͼƬ
	if (!image.data)//�ж�ͼƬ�Ƿ���سɹ�
	{
		printf("�޷�����ͼƬ��\n");
		return -1;
	}

	hog.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 1); //�ڲ���ͼ���ϼ���������� 

	//����foundѰ��û�б�Ƕ�׵ĳ�����
	for (int i = 0; i < regions.size(); i++)
	{
		Rect r = regions[i];

		int j = 0;
		for (; j < regions.size(); j++)
		{
			//���ʱǶ�׵ľ��Ƴ�ѭ��
			if (j != i && (r & regions[j]) == r)
				break;
		}
		if (j == regions.size()){
			regionsRect.push_back(r);
		}
	}

	for (size_t i = 0; i < regionsRect.size(); i++)
	{
		rectangle(image, regionsRect[i], Scalar(0, 0, 255), 2);
	}

	Mat small_image;
	resize(image, small_image, Size(), 0.5, 0.5);//������Ҫ��ͼƬresizeΪԭ����1/4

	int ROIx, ROIy, ROIw, ROIh;//ROI�����Ͻǵ�����λ��(ROIx,ROIy)�����ROIw���߶�ROIy
	ROIx = 51;
	ROIy = 62;
	ROIw = 634;
	ROIh = 100;
	Rect ROIRect = { ROIx, ROIy, ROIw, ROIh };//����ROI,����ƥ���ʱ��
	//Mat imageROI = small_image(ROIRect);
	Mat imageROI = Mat(small_image,ROIRect);

	Mat imageROI_gray;
	cvtColor(imageROI, imageROI_gray, CV_BGR2GRAY);
	equalizeHist(imageROI_gray, imageROI_gray);//ͼ��ֱ��ͼ���⻯����ǿͼ��Աȶ�

	double t = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
	vector<Rect> faces;
	frontalface_cascade.detectMultiScale(imageROI_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	//�����������������������ο�
	t = (double)cvGetTickCount() - t;//��ü�����������ʱ�䲢���
	printf("detection time = %gms/n", t / ((double)cvGetTickFrequency()*1000.));

	for (int i = 0; i < faces.size(); i++)//ѭ�����������ľ��ο�
	{
		rectangle(small_image, Rect{ ROIx + faces[i].x, ROIy + faces[i].y,faces[i].width,faces[i].height }, Scalar(255, 255, 0), 3, 8, 0);
		//rectangle(imageROI, Rect{ faces[i].x, faces[i].y, faces[i].width, faces[i].height }, Scalar(255, 0, 255), 3, 8, 0);//����Ȥ�����ϻ��Ͳ��ü���Ȥ��������Ͻ�������
		//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
		//Point center(ROIx + faces[i].x + faces[i].width*0.5, ROIy + faces[i].y + faces[i].height*0.5);//��Բ���ĵ�����Ҫ����ROI��������Ͻ������
		//ellipse(small_image, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 2, 8, 0);
		//����Բ�������
	}

	imshow("result", small_image);
	//imwrite("�����.jpg",image);//����ͼ�񣬲�֪Ϊ�α����ͼƬ�ܴ󣬶��Ҵ򲻿�
	waitKey(0);
	return 0;
}

//
////ʹ��opencv1д�Ĵ��� 
//#include "stdafx.h"
//#include "cv.h"  
//#include "highgui.h" 
//
//using namespace std;
//using namespace cv;
//static CvMemStorage* storage = 0;
//static CvHaarClassifierCascade* cascade = 0;
//
//const char* cascade_name =
//"C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";/* "haarcascade_profileface.xml";*/
//
//int _tmain(int argc, _TCHAR* argv[])
//{
//	int i;
//	CvPoint pt1, pt2;
//	double scale = 1.0;
//	CvCapture* capture = 0;
//	
//	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);//���ļ���װ��ѵ���õ�Haar����������
//
//	if (!cascade)//�ж�Haar���������Ƿ�ɹ�
//	{
//		fprintf(stderr, "ERROR: Could not load classifier cascade/n");
//		//fprintf( stderr,  
//		//"Usage: facedetect --cascade=/"<cascade_path>"/[filename|camera_index]/n" );  
//		return -1;
//	}
//	storage = cvCreateMemStorage(0);
//
//	cvNamedWindow("result", 1);
//
//	const char* filename = "face.jpg";//����ͼƬ
//	IplImage* image1 = cvLoadImage(filename);
//	if (image1)//�ж�ͼƬ�Ƿ���سɹ�
//	{
//		IplImage* image = cvCreateImage(cvSize(cvRound((image1->width) / 2), cvRound((image1->height) / 2)), 8, 3);
//		cvResize(image1, image, CV_INTER_LINEAR);//��ͼƬresizeΪԭ����1/4	
//		CvRect roi = { 51, 62, 634, 100 };//����ROI,����ƥ���ʱ��
//		cvSetImageROI(image, roi);
//		//�Ƿ�ͼƬת��Ϊ�Ҷ�ͼ���м�����Լ��ʱ��û�ж���Ӱ��
//		//������ROI��ʹ��cvCvtColor��ͼƬת��Ϊ�Ҷ�ͼʱ������ʹ��ԭͼ�Ĵ�С��IplImage* gray = cvCreateImage(cvSize(inmage->width, image->height), 8, 1);������Ȼ������ROI��ͼƬ��С����û��
//		//IplImage* gray = cvCreateImage(cvSize(634, 100), 8, 1);		
//		//cvCvtColor(image, gray, CV_BGR2GRAY);
//		////cvShowImage("gray1", gray);
//		//cvEqualizeHist(gray, gray);
//		////cvShowImage("gray2", gray);
//		double t = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
//		//CvSeq* faces = cvHaarDetectObjects(gray, cascade, storage, 1.1, 2, 0, cvSize(20, 20));//�����������������������ο�
//		CvSeq* faces = cvHaarDetectObjects(image, cascade, storage, 1.1, 2, 0, cvSize(20, 20));//�����������������������ο�
//		t = (double)cvGetTickCount() - t;//��ü�����������ʱ�䲢���
//		printf("detection time = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
//
//		//for (i = 0; i < (faces ? faces->total : 0); i++)//ԭ������������д�ģ���������ʲô��
//		for (i = 0; i < faces->total; i++)//ѭ�����������ľ��ο�
//		{
//			CvRect* r = (CvRect*)cvGetSeqElem(faces, i);//��ü�⵽��һϵ������������ο�
//			//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
//			cvRectangle(image, CvPoint{ cvRound((r->x)*scale), cvRound((r->y)*scale) },\
//				CvPoint{ cvRound((r->x + r->width)*scale), cvRound((r->y + r->height)*scale) }, \
//				CV_RGB(255, 0, 0), 3, 8, 0);
//		}
//		cvResetImageROI(image);//�ͷŻ��ڸ����ľ�������ͼ���ROI���������ͼ���һ�в������Ƕ�ROI������ʾͼƬ��Ҳֻ��ʾROI����
//		cvShowImage("result", image);
//		//cvSave("�����.jpg",image);//����ͼ�񣬲�֪Ϊ�α����ͼƬ�ܴ󣬶��Ҵ򲻿�
//		cvWaitKey(0);
//		cvReleaseImage(&image);
//	}
//
//	cvDestroyWindow("result");
//	cvWaitKey(0);
//	return 0;
//}
//
//////ԭ�����ж�����һϵ����ɫ��ʹ��ͬ��ɫ���������������
////static CvScalar colors[] =
////{
////	{ 0, 0, 255 },
////	{ { 0, 128, 255 } },
////	{ { 0, 255, 255 } },
////	{ { 0, 255, 0 } },
////	{ { 255, 128, 0 } },
////	{ { 255, 255, 0 } },
////	{ { 255, 0, 0 } },
////	{ { 255, 0, 255 } }
////};
////colors[i % 8]    //����ɫ�ĵط�������
//
//////ԭ���̽����ο�����Բ��
////CvRect* r = (CvRect*)cvGetSeqElem(faces, i);
////
////CvPoint center;
////int radius;
////center.x = cvRound((r->x + r->width*0.5)*scale);//���ε����Ͻ�x���Ͼ��ο��ȵ�һ�룬�ٳ���scale(ԭ����С�ĳߴ�)��(ͼ��Ӧ���������Ͻ�Ϊ������зŴ����С)
////center.y = cvRound((r->y + r->height*0.5)*scale);
////radius = cvRound((r->width + r->height)*0.25*scale);//Բ�İ뾶����Ϊ���ο�Ŀ�Ⱥ͸߶�֮�͵�1/4,Ȼ���ٳ���scale
////cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );  

//opencv�ṩ�ļ���������۾��ĳ��� 
//#include "stdafx.h"
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
///** Function Headers */
//void detectAndDisplay(Mat frame);
//
///** Global variables */
//String face_cascade_name = "haarcascade_frontalface_alt.xml";
//String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
//string window_name = "Capture - Face detection";
//RNG rng(12345);
//
///** @function main */
//int main(int argc, const char** argv)
//{
//	CvCapture* capture;
//	Mat frame;
//
//	//-- 1. Load the cascades
//	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
//	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };
//
//	//-- 2. Read the video stream
//	capture = cvCaptureFromCAM(-1);
//	if (capture)
//	{
//		while (true)
//		{
//			frame = cvQueryFrame(capture);
//
//			//-- 3. Apply the classifier to the frame
//			if (!frame.empty())
//			{
//				detectAndDisplay(frame);
//			}
//			else
//			{
//				printf(" --(!) No captured frame -- Break!"); break;
//			}
//
//			int c = waitKey(10);
//			if ((char)c == 'c') { break; }
//		}
//	}
//	return 0;
//}
//
///** @function detectAndDisplay */
//void detectAndDisplay(Mat frame)
//{
//	std::vector<Rect> faces;
//	Mat frame_gray;
//
//	cvtColor(frame, frame_gray, CV_BGR2GRAY);
//	equalizeHist(frame_gray, frame_gray);
//
//	//-- Detect faces
//	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//
//	for (size_t i = 0; i < faces.size(); i++)
//	{
//		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
//		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
//
//		Mat faceROI = frame_gray(faces[i]);
//		std::vector<Rect> eyes;
//
//		//-- In each face, detect eyes
//		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//
//		for (size_t j = 0; j < eyes.size(); j++)
//		{
//			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
//			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
//			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
//		}
//	}
//	//-- Show what you got
//	imshow(window_name, frame);
//}


////����ʹ��opencv2д�Ĵ��� û�иĹ����д���
//#include "stdafx.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <vector>
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
//string face_cascade_name = "C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
////CascadeClassifier face_cascade;
//string window_name = "����ʶ��";
//
//void detectAndDisplay(Mat frame);
//
//int main(int argc, char** argv)
//{
//	Mat image;
//	image = imread("face.jpg");
//
//	if (!image.data){
//		printf("[error] û��ͼƬ\n");
//		return -1;
//	}
//
//	detectAndDisplay(image);
//
//	waitKey(0);
//}
//
//void detectAndDisplay(Mat frame)
//{
//
//	CascadeClassifier face_cascade;
//	if (!face_cascade.load(face_cascade_name)){
//		printf("[error] �޷����ؼ����������ļ���\n");
//		//return -1;
//	}
//
//	std::vector<Rect> faces;
//	Mat frame_gray;
//	cvtColor(frame, frame_gray, CV_BGR2GRAY);
//	equalizeHist(frame_gray, frame_gray);
//
//	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//
//	for (int i = 0; i < faces.size(); i++){
//		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
//		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
//	}
//	imshow(window_name, frame);
//}
