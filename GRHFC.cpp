//opencv2��˹������ģ����������ǰ�����ο�10֡����һ��ǰ�����ο��ڵ�������⣬���ڴ˺��30֡�ڸ�������
//

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

	HOGDescriptor hog;//����HOG���󣬲���Ĭ�ϲ���
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//����Ĭ�ϵ����˼�������
	
	Rect trackWindow;//������ٵľ���
	RotatedRect trackBox;//����һ����ת�ľ����������CamShift����
	int hsize = 16;//ÿһάֱ��ͼ�Ĵ�С
	float hranges[] = { 0, 180 };//hranges�ں���ļ���ֱ��ͼ������Ҫ�õ�
	int vmin = 10, vmax = 256, smin = 30;
	const float* phranges = hranges;//
	Mat hsv, hue, mask, hist, backproj;

	VideoCapture capture;
	VideoWriter capsave;// ������Ƶ
	//capture.open(0);
	capture.open("2.mp4");

	if (!capture.isOpened())
	{
		cout << "������ͷʧ��!" << endl;
		return -1;
	}
	//capture >> frame;//��ȡһ֡����Ƶ�ļ�
	//capsave.open("���.mp4", CV_FOURCC('P', 'I', 'M', '1'), 33, frame.size(), 1);
	//capsave << frame;//����һ֡����Ƶ�ļ�
	//if (!capsave.isOpened())//�жϱ�����Ƶ�Ƿ���ȷ��ʼ��
	//{
	//	cout << "������Ƶʧ��!" << endl;
	//	return -1;
	//}
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
					rects.push_back(boundingRect(contours[idx]));//ѹջ���������������Ӿ��Σ�����rects�ᱣ�����е�������Ӿ���
				}
			}
		}

		for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); it++)//�������з�����������Ӿ���
		{
			rectangle(foreground, *it, Scalar(255, 0, 255), 3, 8, 0);//��ǰ������������������Ӿ��ο�
			frameROIs.push_back(Mat(frame, *it));//�洢ǰ����Ȥ����
		}

		vector<Rect> regions;//�洢������˵ı߽�
		vector<vector<Rect>> Allregions;
		vector<Mat> frameregROIs;
		vector<vector<Mat>> AllfraregROIs;
		if ((frameNo - 1) % 10 == 0)//10֡���һ������������ֻ֡��camshift����
		{
			//��ʼ����Ȥ����������
			//for (int ROINo = 0; ROINo < rects.size(); ROINo++)
			for (vector<Mat>::const_iterator itROIs = frameROIs.begin(); itROIs != frameROIs.end(); itROIs++)
			{
				double t1 = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
				//hog.detectMultiScale(frameROIs[ROINo], regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 1);//������ˣ�����Ĭ�ϲ���������
				hog.detectMultiScale(*itROIs, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 1);//������ˣ�����Ĭ�ϲ���������
				Allregions.push_back(regions);//ѹջ����˴μ�⵽������
				t1 = (double)cvGetTickCount() - t1;//��ü�����������ʱ�䲢���
				printf("������˵�ʱ��Ϊ = %gms/n", t1 / ((double)cvGetTickFrequency()*1000.));
				cout << endl;

				//for (int pNo = 0; pNo < int(regions.size()); pNo++)//ѭ���������˵ľ��ο�
				for (vector<Rect>::const_iterator itreg = regions.begin(); itreg != regions.end(); itreg++)//ѭ�������˴μ�⵽�����˵ľ��ο�
				{
					//rectangle(frame, Rect{ rects[ROINo].x + regions[pNo].x, rects[ROINo].y + regions[pNo].y, regions[pNo].width, regions[pNo].height }, Scalar(0, 0, 255), 3, 8, 0);
					//rectangle(*itROIs, *itreg, Scalar(0, 0, 255), 3, 8, 0);
					//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
					frameregROIs.push_back(Mat(frame, *itreg));//�洢������Ȥ����
				}
				AllfraregROIs.push_back(frameregROIs);//���˴μ�⵽������ѹջ
				frameregROIs.clear();//��ռ�⵽�����ˣ��ѽ�����һ�εļ��
			}

			//��ʼ����������������
			Allfaces.clear();//���¼��ʱ��Ҫ���ԭ���洢������
			faces.clear();			
			//for (int AllfraregNo = 0; AllfraregNo < AllfraregROIs.size(); AllfraregNo++)
			for (vector<vector<Mat>>::const_iterator itArROIs = AllfraregROIs.begin(); itArROIs != AllfraregROIs.end(); itArROIs++)
			{
				//for (int fraregROINo = 0; fraregROINo < AllfraregROIs[AllfraregNo].size(); fraregROINo++)
				for (vector<Mat>::const_iterator itregROIs = frameregROIs.begin(); itregROIs != frameregROIs.end(); itregROIs++)
				{
					double t2 = (double)cvGetTickCount();//��ȡϵͳ��ʱ��
					//frontalface_cascade.detectMultiScale(AllfraregROIs[AllfraregNo][fraregROINo], faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));//�������
					frontalface_cascade.detectMultiScale(*itregROIs, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(20, 20));//�������
					
					Allfaces.push_back(faces);//ѹջ����˴μ�⵽������
					t2 = (double)cvGetTickCount() - t2;//��ü�����������ʱ�䲢���
					printf("���������ʱ��Ϊ = %gms/n", t2 / ((double)cvGetTickFrequency()*1000.));
					cout << endl;

					//for (int fNo = 0; fNo < int(faces.size()); fNo++)//ѭ�����������ľ��ο�
					for (vector<Rect>::const_iterator itf = faces.begin(); itf != faces.end(); itf++)
					{
						//rectangle(frame, Rect{ Allregions[AllfraregNo][fraregROINo].x + faces[fNo].x, Allregions[AllfraregNo][fraregROINo].y + faces[fNo].y, faces[fNo].width, faces[fNo].height }, Scalar(255, 0, 255), 3, 8, 0);
						//rectangle(*itregROIs, *itf, Scalar(255, 0, 255), 3, 8, 0);
						//��ͼƬ�ϻ�����������ķ���1��ͼƬ��2�ξ��ο����Ͻǵ㣻3�ξ������½ǵ㣻4�λ������ο����ɫ��5�ξ��ο���ߴ�ϸ��6�����������ͣ�7��������С����λ��
						//����Բ�������
						//Point center(rects[ROINo].x + faces[fNo].x + faces[fNo].width*0.5, rects[ROINo].y + faces[fNo].y + faces[fNo].height*0.5);//��Բ���ĵ�����Ҫ����ROI��������Ͻ������
						//ellipse(frame, center, Size(faces[fNo].width*0.5, faces[fNo].height*0.5), 0, 0, 360, Scalar(0, 0, 255), 3, 8, 0);
					}
				}
			}
		}

		//camshift��������
		for (vector<vector<Rect>>::iterator itallf = Allfaces.begin(); itallf != Allfaces.end(); itallf++)
		{
			for (vector<Rect>::iterator itf = itallf->begin(); itf != itallf->end(); itf++)
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

		//capsave << frame;//������Ƶ֡

		if (waitKey(33) > 0)
		{
			break;
		}
	}

	return 0;
}