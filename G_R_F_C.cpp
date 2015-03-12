//opencv2高斯背景建模，并画出的前景矩形框，10帧进行一次前景矩形框内的人脸检测，并在此后的30帧内跟踪人脸
//
#include "stdafx.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/video/background_segm.hpp>
#include "cv.h"
#include <iostream>
#include <string>
#include <vector>
#include "stdio.h"
using namespace std;
using namespace cv;

char* cascade_name = "haarcascade_frontalface_alt.xml";
//"haarcascade_frontalface_alt_tree.xml";
//"C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
//"CascadeClassifier frontalface_cascade";

int main()
{
	Mat frame;//存储视频帧	
	Mat foreground;//存储前景
	Mat fgdrect;//复制前景，用于检测轮廓
	Mat background;//存储背景

	vector<vector<Point> > contours;//定义存储边界所需的点	
	vector<Vec4i> hierarchy;//定义存储层次的向量

	Rect trackWindow;//定义跟踪的矩形
	RotatedRect trackBox;//定义一个旋转的矩阵类对象，由CamShift返回
	int hsize = 16;//每一维直方图的大小
	float hranges[] = { 0, 180 };//hranges在后面的计算直方图函数中要用到
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//

	Mat hsv, hue, mask, hist, backproj;

	VideoCapture capture;
	VideoWriter capsave;// 保存视频
	capture.open(0);
	//capture.open("MV.flv");

	if (!capture.isOpened())
	{
		cout << "打开摄像头失败!" << endl;
		return -1;
	}
	capture >> frame;//读取一帧的视频文件
	capsave.open("监控.avi", CV_FOURCC('M', 'J', 'P', 'G'), 33, frame.size(), 1);
	capsave << frame;//保存一帧的视频文件
	if (!capsave.isOpened())//判断保存视频是否正确初始化
	{
		cout << "保存视频失败!" << endl;
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

	vector<vector<Rect>> Allfaces;//存储所有的人脸
	vector<Rect> faces;//存储一个兴趣区域内检测到的人脸。这两个参数需要定义在while循环外面，这样才可以使用CamShift跟踪

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
			frameROIs.push_back(Mat(frame, *it));//存储兴趣区域
		}

		//开始在兴趣区域检测人脸
		if ((frameNo - 1) % 10 == 0)//10帧检测一次人脸，其它帧只用camshift跟踪
		{
			Allfaces.clear();//从新检测时需要清楚原来存储的人脸
			faces.clear();
			for (int ROINo = 0; ROINo < rects.size(); ROINo++)
			{
				double t = (double)cvGetTickCount();//获取系统的时间
				frontalface_cascade.detectMultiScale(frameROIs[ROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
				Allfaces.push_back(faces);//压栈保存此次检测到的人脸
				t = (double)cvGetTickCount() - t;//求得检测人脸所需的时间并输出
				printf("检测人脸的时间为 = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
				cout << endl;

				for (int fNo = 0; fNo < int(faces.size()); fNo++)//循环画出人脸的矩形框
				{
					rectangle(frame, Rect{ rects[ROINo].x + faces[fNo].x, rects[ROINo].y + faces[fNo].y, faces[fNo].width, faces[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
					//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
					//用椭圆标出人脸
					//Point center(rects[ROINo].x + faces[fNo].x + faces[fNo].width*0.5, rects[ROINo].y + faces[fNo].y + faces[fNo].height*0.5);//椭圆中心点坐标要加上ROI区域的左上角坐标点
					//ellipse(frame, center, Size(faces[fNo].width*0.5, faces[fNo].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 3, 8, 0);
				}
			}
		}

		//camshift跟踪人脸
		for (vector<vector<Rect>>::iterator itfall = Allfaces.begin(); itfall != Allfaces.end(); itfall++)
		{
			for (vector<Rect>::iterator itf = itfall->begin(); itf != itfall->end(); itf++)
			{
				cvtColor(frame, hsv, CV_BGR2HSV);//将rgb摄像头帧转化成hsv空间的
				//inRange函数的功能是检查输入数组每个元素大小是否在2个给定数值之间，可以有多通道,mask保存0通道的最小值，也就是h分量
				//这里利用了hsv的3个通道，比较h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)。如果3个通道都在对应的范围内，则
				//mask对应的那个点的值全为1(0xff)，否则为0(0x00).
				inRange(hsv, Scalar(0, smin, 10), Scalar(180, 256, 256), mask);
				int ch[] = { 0, 0 };
				hue.create(hsv.size(), hsv.depth());//hue初始化为与hsv大小深度一样的矩阵，色调的度量是用角度表示的，红绿蓝之间相差120度，反色相差180度
				mixChannels(&hsv, 1, &hue, 1, ch, 1);//将hsv第一个通道(也就是色调)的数复制到hue中，0索引数组

				//此处的构造函数roi用的是Mat hue的矩阵头，且roi的数据指针指向hue，即共用相同的数据，select为其感兴趣的区域
				trackWindow = *itf;
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

				ellipse(frame, trackBox, Scalar(255, 255, 0), 3, CV_AA);//跟踪的时候以椭圆为代表目标
			}
		}

		mog.getBackgroundImage(background);   // 返回当前背景图像
		imshow("video", frame);
		imshow("background", background);
		imshow("foreground", foreground);

		capsave << frame;//保存视频帧

		if (waitKey(33) > 0)
		{
			break;
		}
	}

	return 0;
}




//opencv2高斯背景建模，并用opencv1画出最大的前景矩形框，30帧进行一次前景矩形框内的人脸检测，并在此后的30帧内跟踪人脸
//#include "stdafx.h"
//#include "cv.h"    
//#include "highgui.h"  
//#include <opencv2\core\core.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/video/background_segm.hpp>
//#include <iostream>
//#include <string>
//#include <vector>
//#include "stdio.h"
//using namespace std;
//using namespace cv;
//
//char* cascade_name =
//"C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
//CascadeClassifier frontalface_cascade;
//
//int main()
//{
//	Rect trackWindow;//定义跟踪的矩形
//	RotatedRect trackBox;//定义一个旋转的矩阵类对象
//	int hsize = 16;//每一维直方图的大小
//	float hranges[] = { 0, 180 };//hranges在后面的计算直方图函数中要用到
//	int vmin = 10, vmax = 256, smin = 30;
//	const float* phranges = hranges;//
//	Mat  image, gray, hsv, hue, mask, hist, backproj;
//
//	Mat frame;
//	Mat frameROI;
//	Mat foreground;
//	Mat fgdrect;
//	Mat background;
//	Rect rect;//定义外接矩形，由boundingRect函数返回
//	vector<Rect> faces;//只能放在while循环外面，否则会出错
//
//	VideoCapture capture;
//	capture.open(0);
//
//	if (!capture.isOpened())
//	{
//		cout << "read video failure" << endl;
//		return -1;
//	}
//
//	//CascadeClassifier frontalface_cascade;
//	//从文件中装载训练好的Haar级联分类器
//	if (!frontalface_cascade.load(cascade_name))//判断Haar特征加载是否成功
//	{
//		printf("无法加载级联分类器文件！\n");
//		return -1;
//	}
//
//	BackgroundSubtractorMOG2 mog;	//cv::BackgroundSubtractorMOG mog;
//
//	long frameNo = 0;
//
//	while (capture.read(frame))
//	{
//		++frameNo;
//		cout << "第" << frameNo << "帧" << endl;//输出当前的帧数
//		IplImage frame1 = IplImage(frame);
//		// 运动前景检测，并更新背景
//		mog(frame, foreground, 0.01);
//		//去除噪声
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//膨胀，3*3的element，迭代次数为niters
//		erode(foreground, foreground, Mat(), Point(-1, -1), 1);//腐蚀
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);
//
//		foreground.copyTo(fgdrect);
//		IplImage img = IplImage(fgdrect);
//		IplImage* img_temp = cvCreateImage(cvGetSize(&img), 8, 1);
//
//		CvMemStorage* mem_storage = cvCreateMemStorage(0);
//		CvSeq *first_contour = NULL, *c = NULL;
//
//		cvThreshold(&img, &img, 128, 255, CV_THRESH_BINARY);
//		img_temp = cvCloneImage(&img);
//		cvFindContours(img_temp, mem_storage, &first_contour, sizeof(CvContour), CV_RETR_EXTERNAL);          //#1 需更改区域    
//		cvZero(img_temp);
//
//		c = first_contour;//存储最大轮廓面积的索引
//		double area = 0;//存储最大轮廓的面积
//		for (; first_contour != 0; first_contour = first_contour->h_next)//遍历所有的轮廓
//		{
//			if (fabs(cvContourArea(first_contour)) > area)
//			{
//				c = first_contour;
//				area = fabs(cvContourArea(first_contour));
//			}
//		}
//		//在原图像上标出前景外框
//		if (c != NULL)
//		{
//			rect = cvBoundingRect(c, 1);
//			//cvRectangle(&frame1, cvPoint(rect.x, rect.y), cvPoint((rect.x + rect.width), (rect.y + rect.height)), CV_RGB(0, 255, 255), 2, 8, 0);
//			cvRectangle(&img, cvPoint(rect.x, rect.y), cvPoint((rect.x + rect.width), (rect.y + rect.height)), CV_RGB(0, 255, 255), 2, 8, 0);
//		}
//
//		//开始在兴趣区域检测人脸
//		frameROI = Mat(frame, rect);
//
//		if (frameNo % 30 == 0)//30帧检测一次人脸，其它帧只用camshift跟踪
//		{
//			if (frameROI.data)
//			{
//				double t = (double)cvGetTickCount();//获取系统的时间
//				frontalface_cascade.detectMultiScale(frameROI, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//				t = (double)cvGetTickCount() - t;//求得检测人脸所需的时间并输出
//				printf("检测人脸的时间为 = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
//				cout << endl;
//			}
//
//			if (!faces.empty())
//			{
//				for (int i = 0; i < faces.size(); i++)//循环画出人脸的矩形框
//				{
//					//rectangle(small_image, Rect{ ROIx + faces[i].x, ROIy + faces[i].y,faces[i].width,faces[i].height }, Scalar(255, 0, 255), 3, 8, 0);
//					//在图片上画出人脸区域的方框。1参图片；2参矩形框左上角点；3参矩形右下角点；4参画出矩形框的颜色；5参矩形框的线粗细；6参线条的类型；7参坐标点的小数点位数
//					Point center(rect.x + faces[i].x + faces[i].width*0.5, rect.y + faces[i].y + faces[i].height*0.5);//椭圆中心点坐标要加上ROI区域的左上角坐标点
//					ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 2, 8, 0);
//					//用椭圆标出人脸
//				}
//			}
//		}
//
//		//camshift跟踪人脸
//		if (!faces.empty())//如果trackObject=1且检测到人脸则进入if函数
//		{
//			for (int i = 0; i < faces.size(); i++)
//			{
//				cvtColor(frame, hsv, CV_BGR2HSV);//将rgb摄像头帧转化成hsv空间的
//				//inRange函数的功能是检查输入数组每个元素大小是否在2个给定数值之间，可以有多通道,mask保存0通道的最小值，也就是h分量
//				//这里利用了hsv的3个通道，比较h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)。如果3个通道都在对应的范围内，则
//				//mask对应的那个点的值全为1(0xff)，否则为0(0x00).
//				inRange(hsv, Scalar(0, smin, 10), Scalar(180, 256, 256), mask);
//				int ch[] = { 0, 0 };
//				hue.create(hsv.size(), hsv.depth());//hue初始化为与hsv大小深度一样的矩阵，色调的度量是用角度表示的，红绿蓝之间相差120度，反色相差180度
//				mixChannels(&hsv, 1, &hue, 1, ch, 1);//将hsv第一个通道(也就是色调)的数复制到hue中，0索引数组
//
//				//此处的构造函数roi用的是Mat hue的矩阵头，且roi的数据指针指向hue，即共用相同的数据，select为其感兴趣的区域
//				//Mat roi(hue, selection), maskroi(mask, selection);//mask保存的hsv的最小值
//				trackWindow = faces.at(i);
//				Mat roi(hue, trackWindow), maskroi(mask, trackWindow);//mask保存的hsv的最小值
//
//				//calcHist()函数第一个参数为输入矩阵序列，第2个参数表示输入的矩阵数目，第3个参数表示将被计算直方图维数通道的列表，第4个参数表示可选的掩码函数
//				//第5个参数表示输出直方图，第6个参数表示直方图的维数，第7个参数为每一维直方图数组的大小，第8个参数为每一维直方图bin的边界
//				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);//将roi的0通道计算直方图并通过mask放入hist中，hsize为每一维直方图的大小
//				normalize(hist, hist, 0, 255, CV_MINMAX);//将hist矩阵进行数组范围归一化，都归一化到0~255
//
//				calcBackProject(&hue, 1, 0, hist, backproj, &phranges);//计算直方图的反向投影，计算hue图像0通道直方图hist的反向投影，并让入backproj中
//				backproj &= mask;
//
//				//opencv2.0以后的版本函数命名前没有cv两字了，并且如果函数名是由2个意思的单词片段组成的话，且前面那个片段不够成单词，则第一个字母要
//				//大写，比如Camshift，如果第一个字母是个单词，则小写，比如meanShift，但是第二个字母一定要大写
//				trackBox = CamShift(backproj, trackWindow,               //trackWindow为鼠标选择的区域，TermCriteria为确定迭代终止的准则
//					TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));//CV_TERMCRIT_EPS是通过forest_accuracy,CV_TERMCRIT_ITER
//
//				if (trackWindow.area() <= 1)                                                  //是通过max_num_of_trees_in_the_forest  
//				{
//					int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
//					trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
//						trackWindow.x + r, trackWindow.y + r) &
//						Rect(0, 0, cols, rows);//Rect函数为矩阵的偏移和大小，即第一二个参数为矩阵的左上角点坐标，第三四个参数为矩阵的宽和高
//				}
//
//				ellipse(frame, trackBox, Scalar(255, 0, 0), 3, CV_AA);//跟踪的时候以椭圆为代表目标				
//			}
//		}
//
//		mog.getBackgroundImage(background);
//		cvShowImage("前景轮廓提取", &frame1);
//		cvShowImage("前景", &img);
//		imshow("背景", background);
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
