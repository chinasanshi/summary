//opencv2��˹������ģ����������ǰ�����ο�10֡����һ��ǰ�����ο��ڵ�������⣬���ڴ˺��30֡�ڸ�������
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
	Mat frame;//�洢��Ƶ֡	
	Mat foreground;//�洢ǰ��
	Mat fgdrect;//����ǰ�������ڼ������
	Mat background;//�洢����

	vector<vector<Point> > contours;//����洢�߽�����ĵ�	
	vector<Vec4i> hierarchy;//����洢��ε�����

	Rect trackWindow;//������ٵľ���
	RotatedRect trackBox;//����һ����ת�ľ����������CamShift����
	int hsize = 16;//ÿһάֱ��ͼ�Ĵ�С
	float hranges[] = { 0, 180 };//hranges�ں���ļ���ֱ��ͼ������Ҫ�õ�
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//

	Mat hsv, hue, mask, hist, backproj;

	VideoCapture capture;
	VideoWriter capsave;// ������Ƶ
	capture.open(0);
	//capture.open("MV.flv");

	if (!capture.isOpened())
	{
		cout << "������ͷʧ��!" << endl;
		return -1;
	}
	capture >> frame;//��ȡһ֡����Ƶ�ļ�
	capsave.open("���.avi", CV_FOURCC('M', 'J', 'P', 'G'), 33, frame.size(), 1);
	capsave << frame;//����һ֡����Ƶ�ļ�
	if (!capsave.isOpened())//�жϱ�����Ƶ�Ƿ���ȷ��ʼ��
	{
		cout << "������Ƶʧ��!" << endl;
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

	vector<vector<Rect>> Allfaces;//�洢���е�����
	vector<Rect> faces;//�洢һ����Ȥ�����ڼ�⵽��������������������Ҫ������whileѭ�����棬�����ſ���ʹ��CamShift����

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
			frameROIs.push_back(Mat(frame, *it));//�洢��Ȥ����
		}

		//��ʼ����Ȥ����������
		if ((frameNo - 1) % 10 == 0)//10֡���һ������������ֻ֡��camshift����
		{
			Allfaces.clear();//���¼��ʱ��Ҫ���ԭ���洢������
			faces.clear();
			for (int ROINo = 0; ROINo < rects.size(); ROINo++)
			{
				double t = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
				frontalface_cascade.detectMultiScale(frameROIs[ROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));
				Allfaces.push_back(faces);//ѹջ����˴μ�⵽������
				t = (double)cvGetTickCount() - t;//��ü�����������ʱ�䲢���
				printf("���������ʱ��Ϊ = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
				cout << endl;

				for (int fNo = 0; fNo < int(faces.size()); fNo++)//ѭ�����������ľ��ο�
				{
					rectangle(frame, Rect{ rects[ROINo].x + faces[fNo].x, rects[ROINo].y + faces[fNo].y, faces[fNo].width, faces[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
					//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
					//����Բ�������
					//Point center(rects[ROINo].x + faces[fNo].x + faces[fNo].width*0.5, rects[ROINo].y + faces[fNo].y + faces[fNo].height*0.5);//��Բ���ĵ�����Ҫ����ROI��������Ͻ������
					//ellipse(frame, center, Size(faces[fNo].width*0.5, faces[fNo].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 3, 8, 0);
				}
			}
		}

		//camshift��������
		for (vector<vector<Rect>>::iterator itfall = Allfaces.begin(); itfall != Allfaces.end(); itfall++)
		{
			for (vector<Rect>::iterator itf = itfall->begin(); itf != itfall->end(); itf++)
			{
				cvtColor(frame, hsv, CV_BGR2HSV);//��rgb����ͷ֡ת����hsv�ռ��
				//inRange�����Ĺ����Ǽ����������ÿ��Ԫ�ش�С�Ƿ���2��������ֵ֮�䣬�����ж�ͨ��,mask����0ͨ������Сֵ��Ҳ����h����
				//����������hsv��3��ͨ�����Ƚ�h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)�����3��ͨ�����ڶ�Ӧ�ķ�Χ�ڣ���
				//mask��Ӧ���Ǹ����ֵȫΪ1(0xff)������Ϊ0(0x00).
				inRange(hsv, Scalar(0, smin, 10), Scalar(180, 256, 256), mask);
				int ch[] = { 0, 0 };
				hue.create(hsv.size(), hsv.depth());//hue��ʼ��Ϊ��hsv��С���һ���ľ���ɫ���Ķ������ýǶȱ�ʾ�ģ�������֮�����120�ȣ���ɫ���180��
				mixChannels(&hsv, 1, &hue, 1, ch, 1);//��hsv��һ��ͨ��(Ҳ����ɫ��)�������Ƶ�hue�У�0��������

				//�˴��Ĺ��캯��roi�õ���Mat hue�ľ���ͷ����roi������ָ��ָ��hue����������ͬ�����ݣ�selectΪ�����Ȥ������
				trackWindow = *itf;
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

				ellipse(frame, trackBox, Scalar(255, 255, 0), 3, CV_AA);//���ٵ�ʱ������ԲΪ����Ŀ��
			}
		}

		mog.getBackgroundImage(background);   // ���ص�ǰ����ͼ��
		imshow("video", frame);
		imshow("background", background);
		imshow("foreground", foreground);

		capsave << frame;//������Ƶ֡

		if (waitKey(33) > 0)
		{
			break;
		}
	}

	return 0;
}




//opencv2��˹������ģ������opencv1��������ǰ�����ο�30֡����һ��ǰ�����ο��ڵ�������⣬���ڴ˺��30֡�ڸ�������
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
//	Rect trackWindow;//������ٵľ���
//	RotatedRect trackBox;//����һ����ת�ľ��������
//	int hsize = 16;//ÿһάֱ��ͼ�Ĵ�С
//	float hranges[] = { 0, 180 };//hranges�ں���ļ���ֱ��ͼ������Ҫ�õ�
//	int vmin = 10, vmax = 256, smin = 30;
//	const float* phranges = hranges;//
//	Mat  image, gray, hsv, hue, mask, hist, backproj;
//
//	Mat frame;
//	Mat frameROI;
//	Mat foreground;
//	Mat fgdrect;
//	Mat background;
//	Rect rect;//������Ӿ��Σ���boundingRect��������
//	vector<Rect> faces;//ֻ�ܷ���whileѭ�����棬��������
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
//	//���ļ���װ��ѵ���õ�Haar����������
//	if (!frontalface_cascade.load(cascade_name))//�ж�Haar���������Ƿ�ɹ�
//	{
//		printf("�޷����ؼ����������ļ���\n");
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
//		cout << "��" << frameNo << "֡" << endl;//�����ǰ��֡��
//		IplImage frame1 = IplImage(frame);
//		// �˶�ǰ����⣬�����±���
//		mog(frame, foreground, 0.01);
//		//ȥ������
//		dilate(foreground, foreground, Mat(), Point(-1, -1), 1);//���ͣ�3*3��element����������Ϊniters
//		erode(foreground, foreground, Mat(), Point(-1, -1), 1);//��ʴ
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
//		cvFindContours(img_temp, mem_storage, &first_contour, sizeof(CvContour), CV_RETR_EXTERNAL);          //#1 ���������    
//		cvZero(img_temp);
//
//		c = first_contour;//�洢����������������
//		double area = 0;//�洢������������
//		for (; first_contour != 0; first_contour = first_contour->h_next)//�������е�����
//		{
//			if (fabs(cvContourArea(first_contour)) > area)
//			{
//				c = first_contour;
//				area = fabs(cvContourArea(first_contour));
//			}
//		}
//		//��ԭͼ���ϱ��ǰ�����
//		if (c != NULL)
//		{
//			rect = cvBoundingRect(c, 1);
//			//cvRectangle(&frame1, cvPoint(rect.x, rect.y), cvPoint((rect.x + rect.width), (rect.y + rect.height)), CV_RGB(0, 255, 255), 2, 8, 0);
//			cvRectangle(&img, cvPoint(rect.x, rect.y), cvPoint((rect.x + rect.width), (rect.y + rect.height)), CV_RGB(0, 255, 255), 2, 8, 0);
//		}
//
//		//��ʼ����Ȥ����������
//		frameROI = Mat(frame, rect);
//
//		if (frameNo % 30 == 0)//30֡���һ������������ֻ֡��camshift����
//		{
//			if (frameROI.data)
//			{
//				double t = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
//				frontalface_cascade.detectMultiScale(frameROI, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//				t = (double)cvGetTickCount() - t;//��ü�����������ʱ�䲢���
//				printf("���������ʱ��Ϊ = %gms/n", t / ((double)cvGetTickFrequency()*1000.));
//				cout << endl;
//			}
//
//			if (!faces.empty())
//			{
//				for (int i = 0; i < faces.size(); i++)//ѭ�����������ľ��ο�
//				{
//					//rectangle(small_image, Rect{ ROIx + faces[i].x, ROIy + faces[i].y,faces[i].width,faces[i].height }, Scalar(255, 0, 255), 3, 8, 0);
//					//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
//					Point center(rect.x + faces[i].x + faces[i].width*0.5, rect.y + faces[i].y + faces[i].height*0.5);//��Բ���ĵ�����Ҫ����ROI��������Ͻ������
//					ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 2, 8, 0);
//					//����Բ�������
//				}
//			}
//		}
//
//		//camshift��������
//		if (!faces.empty())//���trackObject=1�Ҽ�⵽���������if����
//		{
//			for (int i = 0; i < faces.size(); i++)
//			{
//				cvtColor(frame, hsv, CV_BGR2HSV);//��rgb����ͷ֡ת����hsv�ռ��
//				//inRange�����Ĺ����Ǽ����������ÿ��Ԫ�ش�С�Ƿ���2��������ֵ֮�䣬�����ж�ͨ��,mask����0ͨ������Сֵ��Ҳ����h����
//				//����������hsv��3��ͨ�����Ƚ�h,0~180,s,smin~256,v,min(vmin,vmax),max(vmin,vmax)�����3��ͨ�����ڶ�Ӧ�ķ�Χ�ڣ���
//				//mask��Ӧ���Ǹ����ֵȫΪ1(0xff)������Ϊ0(0x00).
//				inRange(hsv, Scalar(0, smin, 10), Scalar(180, 256, 256), mask);
//				int ch[] = { 0, 0 };
//				hue.create(hsv.size(), hsv.depth());//hue��ʼ��Ϊ��hsv��С���һ���ľ���ɫ���Ķ������ýǶȱ�ʾ�ģ�������֮�����120�ȣ���ɫ���180��
//				mixChannels(&hsv, 1, &hue, 1, ch, 1);//��hsv��һ��ͨ��(Ҳ����ɫ��)�������Ƶ�hue�У�0��������
//
//				//�˴��Ĺ��캯��roi�õ���Mat hue�ľ���ͷ����roi������ָ��ָ��hue����������ͬ�����ݣ�selectΪ�����Ȥ������
//				//Mat roi(hue, selection), maskroi(mask, selection);//mask�����hsv����Сֵ
//				trackWindow = faces.at(i);
//				Mat roi(hue, trackWindow), maskroi(mask, trackWindow);//mask�����hsv����Сֵ
//
//				//calcHist()������һ������Ϊ����������У���2��������ʾ����ľ�����Ŀ����3��������ʾ��������ֱ��ͼά��ͨ�����б���4��������ʾ��ѡ�����뺯��
//				//��5��������ʾ���ֱ��ͼ����6��������ʾֱ��ͼ��ά������7������Ϊÿһάֱ��ͼ����Ĵ�С����8������Ϊÿһάֱ��ͼbin�ı߽�
//				calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);//��roi��0ͨ������ֱ��ͼ��ͨ��mask����hist�У�hsizeΪÿһάֱ��ͼ�Ĵ�С
//				normalize(hist, hist, 0, 255, CV_MINMAX);//��hist����������鷶Χ��һ��������һ����0~255
//
//				calcBackProject(&hue, 1, 0, hist, backproj, &phranges);//����ֱ��ͼ�ķ���ͶӰ������hueͼ��0ͨ��ֱ��ͼhist�ķ���ͶӰ��������backproj��
//				backproj &= mask;
//
//				//opencv2.0�Ժ�İ汾��������ǰû��cv�����ˣ������������������2����˼�ĵ���Ƭ����ɵĻ�����ǰ���Ǹ�Ƭ�β����ɵ��ʣ����һ����ĸҪ
//				//��д������Camshift�������һ����ĸ�Ǹ����ʣ���Сд������meanShift�����ǵڶ�����ĸһ��Ҫ��д
//				trackBox = CamShift(backproj, trackWindow,               //trackWindowΪ���ѡ�������TermCriteriaΪȷ��������ֹ��׼��
//					TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));//CV_TERMCRIT_EPS��ͨ��forest_accuracy,CV_TERMCRIT_ITER
//
//				if (trackWindow.area() <= 1)                                                  //��ͨ��max_num_of_trees_in_the_forest  
//				{
//					int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5) / 6;
//					trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
//						trackWindow.x + r, trackWindow.y + r) &
//						Rect(0, 0, cols, rows);//Rect����Ϊ�����ƫ�ƺʹ�С������һ��������Ϊ��������Ͻǵ����꣬�����ĸ�����Ϊ����Ŀ�͸�
//				}
//
//				ellipse(frame, trackBox, Scalar(255, 0, 0), 3, CV_AA);//���ٵ�ʱ������ԲΪ����Ŀ��				
//			}
//		}
//
//		mog.getBackgroundImage(background);
//		cvShowImage("ǰ��������ȡ", &frame1);
//		cvShowImage("ǰ��", &img);
//		imshow("����", background);
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
