// facedect.cpp : Defines the entry point for the console application.
//
//自己使用opencv2改写的代码 
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
	//从文件中装载训练好的Haar级联分类器
	if (!frontalface_cascade.load(cascade_name))//判断Haar特征加载是否成功
	{
		printf("无法加载级联分类器文件！\n");
		return -1;
	}
	HOGDescriptor hog;//定义HOG对象,采用默认参数
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector()); //设置SVM分类器，采用已经训练好的行人检测分类器
	vector<Rect> regions,regionsRect;

	namedWindow("result", 1);

	Mat image;
	image = imread("face.jpg");//载入图片
	if (!image.data)//判断图片是否加载成功
	{
		printf("无法加载图片！\n");
		return -1;
	}

	hog.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 1); //在测试图像上检测行人区域 

	//遍历found寻找没有被嵌套的长方形
	for (int i = 0; i < regions.size(); i++)
	{
		Rect r = regions[i];

		int j = 0;
		for (; j < regions.size(); j++)
		{
			//如果时嵌套的就推出循环
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
	resize(image, small_image, Size(), 0.5, 0.5);//根据需要将图片resize为原来的1/4

	int ROIx, ROIy, ROIw, ROIh;//ROI的左上角点坐标位置(ROIx,ROIy)，宽度ROIw，高度ROIy
	ROIx = 51;
	ROIy = 62;
	ROIw = 634;
	ROIh = 100;
	Rect ROIRect = { ROIx, ROIy, ROIw, ROIh };//设置ROI,减少匹配的时间
	//Mat imageROI = small_image(ROIRect);
	Mat imageROI = Mat(small_image,ROIRect);

	Mat imageROI_gray;
	cvtColor(imageROI, imageROI_gray, CV_BGR2GRAY);
	equalizeHist(imageROI_gray, imageROI_gray);//图像直方图均衡化，增强图像对比度

	double t = (double)cvGetTickCount();//获取系统的时间
	vector<Rect> faces;
	frontalface_cascade.detectMultiScale(imageROI_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	//检测人脸，返回人脸区域矩形框
	t = (double)cvGetTickCount() - t;//求得检测人脸所需的时间并输出
	printf("detection time = %gms/n", t / ((double)cvGetTickFrequency()*1000.));

	for (int i = 0; i < faces.size(); i++)//循环画出人脸的矩形框
	{
		rectangle(small_image, Rect{ ROIx + faces[i].x, ROIy + faces[i].y,faces[i].width,faces[i].height }, Scalar(255, 255, 0), 3, 8, 0);
		//rectangle(imageROI, Rect{ faces[i].x, faces[i].y, faces[i].width, faces[i].height }, Scalar(255, 0, 255), 3, 8, 0);//在兴趣区域上画就不用加兴趣区域的左上角坐标了
		//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
		//Point center(ROIx + faces[i].x + faces[i].width*0.5, ROIy + faces[i].y + faces[i].height*0.5);//椭圆中心点坐标要加上ROI区域的左上角坐标点
		//ellipse(small_image, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 2, 8, 0);
		//用椭圆标出人脸
	}

	imshow("result", small_image);
	//imwrite("标出后.jpg",image);//保存图像，不知为何保存的图片很大，而且打不开
	waitKey(0);
	return 0;
}

//
////使用opencv1写的代码 
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
//	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);//从文件中装载训练好的Haar级联分类器
//
//	if (!cascade)//判断Haar特征加载是否成功
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
//	const char* filename = "face.jpg";//载入图片
//	IplImage* image1 = cvLoadImage(filename);
//	if (image1)//判断图片是否加载成功
//	{
//		IplImage* image = cvCreateImage(cvSize(cvRound((image1->width) / 2), cvRound((image1->height) / 2)), 8, 3);
//		cvResize(image1, image, CV_INTER_LINEAR);//将图片resize为原来的1/4	
//		CvRect roi = { 51, 62, 634, 100 };//设置ROI,减少匹配的时间
//		cvSetImageROI(image, roi);
//		//是否将图片转换为灰度图进行检测好像对检测时间没有多大的影响
//		//设置了ROI，使用cvCvtColor将图片转换为灰度图时不可以使用原图的大小（IplImage* gray = cvCreateImage(cvSize(inmage->width, image->height), 8, 1);），虽然设置了ROI但图片大小好像没变
//		//IplImage* gray = cvCreateImage(cvSize(634, 100), 8, 1);		
//		//cvCvtColor(image, gray, CV_BGR2GRAY);
//		////cvShowImage("gray1", gray);
//		//cvEqualizeHist(gray, gray);
//		////cvShowImage("gray2", gray);
//		double t = (double)cvGetTickCount();//获取系统的时间
//		//CvSeq* faces = cvHaarDetectObjects(gray, cascade, storage, 1.1, 2, 0, cvSize(20, 20));//检测人脸，返回人脸区域矩形框
//		CvSeq* faces = cvHaarDetectObjects(image, cascade, storage, 1.1, 2, 0, cvSize(20, 20));//检测人脸，返回人脸区域矩形框
//		t = (double)cvGetTickCount() - t;//求得检测人脸所需的时间并输出
//		printf("detection time = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
//
//		//for (i = 0; i < (faces ? faces->total : 0); i++)//原例程中是这样写的，不明白有什么用
//		for (i = 0; i < faces->total; i++)//循环画出人脸的矩形框
//		{
//			CvRect* r = (CvRect*)cvGetSeqElem(faces, i);//获得检测到的一系列人脸区域矩形框
//			//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
//			cvRectangle(image, CvPoint{ cvRound((r->x)*scale), cvRound((r->y)*scale) },\
//				CvPoint{ cvRound((r->x + r->width)*scale), cvRound((r->y + r->height)*scale) }, \
//				CV_RGB(255, 0, 0), 3, 8, 0);
//		}
//		cvResetImageROI(image);//释放基于给定的矩形设置图像的ROI，否则针对图像的一切操作都是对ROI区域，显示图片将也只显示ROI区域
//		cvShowImage("result", image);
//		//cvSave("标出后.jpg",image);//保存图像，不知为何保存的图片很大，而且打不开
//		cvWaitKey(0);
//		cvReleaseImage(&image);
//	}
//
//	cvDestroyWindow("result");
//	cvWaitKey(0);
//	return 0;
//}
//
//////原例程中定义了一系列颜色，使不同颜色来标记人脸的区域
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
////colors[i % 8]    //在颜色的地方的设置
//
//////原例程将矩形框变成了圆形
////CvRect* r = (CvRect*)cvGetSeqElem(faces, i);
////
////CvPoint center;
////int radius;
////center.x = cvRound((r->x + r->width*0.5)*scale);//矩形的左上角x加上矩形框宽度的一半，再乘以scale(原来缩小的尺寸)；(图像应该是以左上角为基点进行放大和缩小)
////center.y = cvRound((r->y + r->height*0.5)*scale);
////radius = cvRound((r->width + r->height)*0.25*scale);//圆的半径设置为矩形框的宽度和高度之和的1/4,然后再乘以scale
////cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );  

//opencv提供的检测人脸及眼睛的程序 
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


////别人使用opencv2写的代码 没有改过来有错误
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
//string window_name = "人脸识别";
//
//void detectAndDisplay(Mat frame);
//
//int main(int argc, char** argv)
//{
//	Mat image;
//	image = imread("face.jpg");
//
//	if (!image.data){
//		printf("[error] 没有图片\n");
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
//		printf("[error] 无法加载级联分类器文件！\n");
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
