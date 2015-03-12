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
	cout << "\n这是基于Haar特征的人脸检测，并使用CamShift进行跟踪的一个demo\n"
		"程序会自动检测人脸并进行跟踪，可以检测并跟踪多个人脸\n"
		"程序会打开摄像头采集视频，开始打开视频的时候会检测一次人脸，若没有检测到或想要重新检测按键‘d’即可\n";

	cout << "\n\n功能键: \n"
		"\tEsc - 退出程序\n"
		"\tc - 停止跟踪\n"
		"\td - 重新检测人脸\n";
}

int main()
{
	help();

	CascadeClassifier frontalface_cascade;
	//从文件中装载训练好的Haar级联分类器
	if (!frontalface_cascade.load(cascade_name))//判断Haar特征加载是否成功
	{
		printf("无法加载级联分类器文件！\n");
		return -1;
	}	

	int camwidth=640, camheight = 480;//定义摄像头图像的宽度和高度
	int detction = 1;//代表是否使用人脸检测函数检测人脸
	int trackObject = 0; //代表是否在跟踪目标
	Rect trackWindow;//定义跟踪的矩形
	RotatedRect trackBox;//定义一个旋转的矩阵类对象
	int hsize = 16;//每一维直方图的大小
	float hranges[] = { 0, 180 };//hranges在后面的计算直方图函数中要用到
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//
	Mat frame, image, gray, hsv, hue, mask, hist, backproj;// histimg = Mat::zeros(200, 320, CV_8UC3), 
	//Mat image, gray;//放在循环内外好像无所谓

	VideoCapture capture;
	capture.open(0);//也可以写成VideoCapture capture(0);
	if (!capture.isOpened())
	{
		printf("打开摄像头失败！\n");
		return -1;
	}
	capture.set(CV_CAP_PROP_FRAME_WIDTH, camwidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, camheight);//设置图像额宽度和高度

	namedWindow("待检测视频");
	vector<Rect> faces;//只能放在while循环外面，否则会出错
	int facenum;
	while (1)
	{
		capture >> frame;//读取视频的下一帧
		if (frame.empty())
		{
			break;
		}
		frame.copyTo(image); 

		if (detction)//初始化为1，所以会先进行一次人脸检测
		{
			detction = 0;//Haar检测完人脸后detction一直保持为0，因此该if函数只能执行一次，除非重新按下字母d
			trackObject = 1;
			cvtColor(image, gray, CV_BGR2GRAY);//将rgb摄像头帧转化成灰度图
			equalizeHist(gray, gray);

			double t = (double)cvGetTickCount();//获取系统的时间
			//检测人脸，返回人脸区域矩形框
			frontalface_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));			
			t = (double)cvGetTickCount() - t;//求得检测人脸所需的时间并输出
			printf("检测人脸的时间为 = %gms/n\n", t / ((double)cvGetTickFrequency()*1000.));
			facenum = faces.size();
		}

		if (trackObject && (!faces.empty()))//如果trackObject=1且检测到人脸则进入if函数
		{
			for (int i = 0; i < facenum; i++)
			{
				cvtColor(image, hsv, CV_BGR2HSV);//将rgb摄像头帧转化成hsv空间的
				//inRange函数的功能是检查输入数组每个元素大小是否在2个给定数值之间，可以有多通道,mask保存0通道的最小值，也就是h分量
				//这里利用了hsv的3个通道，比较h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)。如果3个通道都在对应的范围内，则
				//mask对应的那个点的值全为1(0xff)，否则为0(0x00).
				inRange(hsv, Scalar(0, smin, 10), Scalar(180, 256, 256), mask);
				int ch[] = { 0, 0 };
				hue.create(hsv.size(), hsv.depth());//hue初始化为与hsv大小深度一样的矩阵，色调的度量是用角度表示的，红绿蓝之间相差120度，反色相差180度
				mixChannels(&hsv, 1, &hue, 1, ch, 1);//将hsv第一个通道(也就是色调)的数复制到hue中，0索引数组

				//此处的构造函数roi用的是Mat hue的矩阵头，且roi的数据指针指向hue，即共用相同的数据，select为其感兴趣的区域
				//Mat roi(hue, selection), maskroi(mask, selection);//mask保存的hsv的最小值
				trackWindow = faces.at(i);
				Mat roi(hue, trackWindow), maskroi(mask, trackWindow);//mask保存的hsv的最小值

				//calcHist()函数第一个参数为输入矩阵序列，第2个参数表示输入的矩阵数目，第3个参数表示将被计算直方图维数通道的列表，第4个参数表示可选的掩码函数
				//第5个参数表示输出直方图，第6个参数表示直方图的维数，第7个参数为每一维直方图数组的大小，第8个参数为每一维直方图bin的边界
				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);//将roi的0通道计算直方图并通过mask放入hist中，hsize为每一维直方图的大小
				normalize(hist, hist, 0, 255, CV_MINMAX);//将hist矩阵进行数组范围归一化，都归一化到0~255

				calcBackProject(&hue, 1, 0, hist, backproj, &phranges);//计算直方图的反向投影，计算hue图像0通道直方图hist的反向投影，并让入backproj中
				backproj &= mask;

				//opencv2.0以后的版本函数命名前没有cv两字了，并且如果函数名是由2个意思的单词片段组成的话，且前面那个片段不够成单词，则第一个字母要
				//大写，比如Camshift，如果第一个字母是个单词，则小写，比如meanShift，但是第二个字母一定要大写
				trackBox = CamShift(backproj, trackWindow,               //trackWindow为鼠标选择的区域，TermCriteria为确定迭代终止的准则
					TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));//CV_TERMCRIT_EPS是通过forest_accuracy,CV_TERMCRIT_ITER

				if (trackWindow.area() <= 1)                                                  //是通过max_num_of_trees_in_the_forest  
				{
					int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
					trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
						trackWindow.x + r, trackWindow.y + r) &
						Rect(0, 0, cols, rows);//Rect函数为矩阵的偏移和大小，即第一二个参数为矩阵的左上角点坐标，第三四个参数为矩阵的宽和高
				}

				ellipse(image, trackBox, Scalar(255, 255, 0), 3, CV_AA);//跟踪的时候以椭圆为代表目标				
			}
		}

		imshow("待检测视频", image);

		char c = (char)waitKey(10);

		switch (c)
		{
			case 'c':            //清零trackObject，从而不再跟踪目标对象
				trackObject = 0;
				break;
			case 'd':       //使用Haar特征检测人脸
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


////用到 kalman 滤波进行预测，这样能智能的扩大范围进行 camshift 处理
////先初始化
//const int stateDim = 4;
//const int measureDim = 2;
//const int contralDim = 0;
//KalmanFilter KF;
//KF.init(stateDim, measureDim, contralDim, CV_32F);
//Mat state(stateDim, 1, CV_32F); //源程序没有这句
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
//int ROIx, ROIy, ROIw, ROIh;//ROI的左上角点坐标位置(ROIx,ROIy)，宽度ROIw，高度ROIy
//ROIx = 51;
//ROIy = 62;
//ROIw = 634;
//ROIh = 100;
//Rect ROIRect = { ROIx, ROIy, ROIw, ROIh };//设置ROI,减少匹配的时间
//Mat imageROI = small_image(ROIRect);

//Mat imageROI_gray;
//cvtColor(imageROI, imageROI_gray, CV_BGR2GRAY);
//equalizeHist(imageROI_gray, imageROI_gray);//图像直方图均衡化，增强图像对比度