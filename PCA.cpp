//
//
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <io.h>
//#include <libxml/parser.h>  
//#include <libxml/tree.h>
//#include <fstream>
//#include <sstream>

using namespace cv;
using namespace std;

// ����ͼƬ����������Ӧ��ŵ�����
vector<Mat> faces;
vector<int> labels;

static Mat toGrayscale(Mat src);
Mat facedect(Mat image);
void takphoto();
bool transfer(string fileName);
void predect(Ptr<FaceRecognizer> model);

int main(int argc, const char *argv[])
{
	//takphoto();//�ɼ���������
	transfer("D:\\face\\kyl\\cpp\\PCA\\PCA\\*.jpg");//������·���µ����з������������.jpgͼƬ
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();//����PCAģ��
	model->train(faces, labels);//ѵ��ģ��

	predect(model);//������ͷ��Ԥ������
	
	waitKey(0);

	return 0;
}


Mat facedect(Mat image)//���һ��ͼƬ�������
{
	char* cascade_name = "C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
	CascadeClassifier frontalface_cascade;
	if (!frontalface_cascade.load(cascade_name))//�ж�Haar���������Ƿ�ɹ�
	{
		printf("�޷����ؼ����������ļ���\n");
	}
	vector<Rect> faces;
	Mat face;
	frontalface_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (!faces.empty())
	{
		Mat tempface(image, faces[0]);		
		tempface.copyTo(face);		
	}
	return face;
}
//ֻ����Ϊ�Ҷ�ͼ�����򱨴�Ȼ�󽫻Ҷ�ͼ��һ��
static Mat toGrayscale(Mat src)
{
	// ֻ����Ϊ��ͨ����
	if (src.channels() != 1)
	{
		CV_Error(CV_StsBadArg, "ֻ֧�ֵ�ͨ���ľ���");
	}
	// ���������ع�һ�����ͼƬ
	Mat dst;
	cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}
  
//���������b������ʼ�ɼ����������������f�������������ɼ�������������q�����˳��ɼ�������������ʾ�������������ɼ����
void takphoto()
{
	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << "������ͷʧ�ܣ�" << endl;
	}

	bool beginface = false;//�Ƿ�ʼ�ɼ������ı�־λ

	Mat frame;//������Ƶ֡
	Mat temp;//�ɼ�����ʱ��ʱ���Ƶ�ǰ��Ƶ֡
	Mat face;//����ɼ���������
	//Mat not;//���水λȡ�������Ƶͼ��
	namedWindow("img");
	//vector<Mat> faces;
	//vector<int> labels;
	int label = 0;//��ǰ�����ı�ţ�ÿ�ɼ�һ���Լ�1
	int faceNo = 0;//���浱ǰ�ɼ������ĸ���
	int AllfaceNo = 0;//�ɼ�����������
	long frameNo = 0;//��Ƶ��֡��
	long NowframNo = 0;//���濪ʼ�ɼ�����ʱ����Ƶ֡��

	string name = "f";

	while (capture.read(frame))
	{
		frameNo++;//֡����1
		cout << "��ǰ֡��Ϊ" << frameNo << "; "; //�����ǰ֡��
		//bitwise_not(frame,not);//��frame��λȡ���õ�ȡ�����ͼ��
		//imshow("not", not);//��ʾ��λȡ�����ͼ��
		imshow("img", frame);//��ʾ��Ƶ֡
		char c = waitKey(33);//����֡��
		if (c == 'b')//���������b������ʼ�ɼ�����
		{
			beginface = true;//����־λ��Ϊ��
			label++;//��ʼһ�������ɼ�label�Լ�1����������ͬ����
			name = "f";//ÿ�ο�ʼ�ɼ����������¸����ַ�ֵ
			stringstream ss;//stringstream�������²�ͬ�����ͣ�����s1�����ͣ�Ȼ���³���ͬ������
			ss << label;
			string s1 = ss.str();//��Ϊ�ַ���
			name += s1;//name��Ϊn���ϱ�ţ���n1,n2����	
			name += "_";//name��Ϊn1_,n2_����	
			NowframNo = frameNo;//���浱ǰ��֡���������жϵ�ǰ֡��ʼ���ÿ30֡
			cout << endl << endl << "��ʼ�ɼ����������˱��Ϊ" << label << endl << endl;
		}
		//���㿪ʼ�����������ӵ�ǰʱ�̿�ʼÿ30֡�ɼ�һ������
		if (((beginface) && (frameNo - NowframNo - 30) % 30 == 0))
		{
			frame.copyTo(temp);//���Ƶ�ǰ��Ƶ֡
			face = facedect(temp);//�ڵ�ǰ֡�м������
			if (face.empty())//û�м�⵽��������ʾ����
			{
				cout<< endl << endl << "û�м�⵽���������׼����ͷ��" << endl << endl;
				//break;//��������break��������˳������ɼ�
			}
			else//���м�⵽������������²���
			{
				resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��
				cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
				toGrayscale(face);//��һ��
				faces.push_back(face);//������ѹջ
				labels.push_back(label);//�����ѹջ
				faceNo++;//ÿ����һ��ѭ���������Լ�1��������¼�˴βɼ�����������
				AllfaceNo++;//�����ܹ��ɼ����������ĸ���
				stringstream ss;//stringstream�������²�ͬ�����ͣ�����s1�����ͣ�Ȼ���³���ͬ������
				ss << faceNo;
				string s1 = ss.str();//��Ϊ�ַ���
				name += s1;//name��Ϊn���ϱ�ţ���n1,n2����	
				name += ".jpg";
				namedWindow("face");
				imshow("face", face);//��ʾ�˴βɼ���������			
				imwrite(name, face);
				cout <<endl << endl << "��⵽����������ͼƬ" << name << endl;
				name.pop_back();
				name.pop_back();
				name.pop_back();
				name.pop_back();//����4���ǵ���".jpg"
				name.pop_back();//�������
				if (faceNo > 9)//�����������Ϊ��λ����Ҫ�൯��һ�Σ������������������99���򱣴��ͷ������ֻ����
				{
					name.pop_back();
				}
				cout << "Ŀǰ�ɼ����������ĸ���Ϊ" << faceNo << endl << endl;
			}
		}
		if (c == 'f')//���������f�������������ɼ�����
		{
			beginface = false;//��ֵ��һ�βɼ�
			faceNo -= 1;
			AllfaceNo -= 1;
			faces.pop_back();
			labels.pop_back();//�������һ�βɼ�����������ΪЧ������
			cout << "�ɼ���������ϣ��ܹ��ɼ�����������Ϊ" << faceNo << endl;
			faceNo = 0;//�����������㣬Ϊ��һλ�����ɼ���׼��
		}
		if (c == 'q')//������q�����˳��ɼ�������������ʾ�������������ɼ����
		{
			cout << "���������ɼ���ϣ�" << endl;
			cout << "�ܹ��ɼ�������������Ϊ"<< AllfaceNo << endl;
			break;
		}
	}
	destroyWindow("img");//�ر���Ƶ����
	destroyWindow("face");//�ر�face����
}


//�ṹ��˵��
//struct _finddata_t
//{
//	unsigned attrib;     //�ļ�����
//	time_t time_create;  //�ļ�����ʱ��
//	time_t time_access;  //�ļ���һ�η���ʱ��
//	time_t time_write;   //�ļ���һ���޸�ʱ��
//	_fsize_t size;  //�ļ��ֽ���
//	char name[_MAX_FNAME]; //�ļ���
//};
////��FileName��������ƥ�䵱ǰĿ¼��һ���ļ�
//_findfirst(_In_ const char * FileName, _Out_ struct _finddata64i32_t * _FindData);
////��FileName��������ƥ�䵱ǰĿ¼��һ���ļ�
//_findnext(_In_ intptr_t _FindHandle, _Out_ struct _finddata64i32_t * _FindData);
////�ر�_findfirst���ص��ļ����
//_findclose(_In_ intptr_t _FindHandle);

bool transfer(string fileName)//filename������ͨ�����'��'����һ���ַ���'*'����0�������ַ�
{
	int jpgNum = 0;//��¼�ܹ��ҵ���ͼƬ�ĸ���
	char labelchar[10]="\0";//�����ͼƬ����������ȡ���ı��
	int label = 0;
	_finddata_t fileInfo;//����ṹ��
	long handle = _findfirst(fileName.c_str(), &fileInfo);//Ѱ�ҵ�һ��ͼƬ�ļ�

	if (handle == -1L)//���û���ҵ��ļ���������-1L
	{
		cerr << "δ�ҵ�ͼƬ�ļ���" << endl;
		return false;
	}

	do
	{	
		if (fileInfo.name[0] != 'f')//���ͼƬ���Ʋ�����f��ͷ�򲻷���������������һ�β���
		{
			continue;//������break�������ֱ���˳�do-whileѭ��
		}
		for (int i = 0; i < strlen(fileInfo.name); i++)
		{
			if (fileInfo.name[i + 1] == '_')//�����»������˳�forѭ��
			{
				labelchar[i] = '\0';
				break;
			}
			labelchar[i] = fileInfo.name[i + 1];//ͼƬ���ֵĵ�һ����ĸΪf�����ñ���
		}
		label = atoi(labelchar);//��char�ͱ��ת����int��
		labels.push_back(label);//�����ѹջ����������
		faces.push_back(imread(fileInfo.name,0));//���ҵ���ͼƬ�ԻҶ���ʽѹջ����faces�����򲻻��н��
		jpgNum++;//ͼƬ����1
		cout << fileInfo.name << endl;//����ҵ���ͼƬ������
	} while (_findnext(handle, &fileInfo) == 0);//Ѱ����һ��ͼƬ
	cout << " ��Ч .jpg ͼƬ�ļ��ĸ���Ϊ:  " << jpgNum << endl;//����ҵ���ͼƬ�ĸ���

	return true;
}

//����p��Ԥ������������q�˳�����Ԥ��
void predect(Ptr<FaceRecognizer> model)
{
	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << "������ͷʧ�ܣ�" << endl;
	}
	Mat frame;//������Ƶ֡
	Mat temp;
	Mat face;
	namedWindow("����");
	bool ispre = false;
	int predictedLabel;
	while (capture.read(frame))
	{
		char c = waitKey(33);//����֡��
		imshow("predect", frame);//��ʾ��Ƶ
		if (c == 'p')//����p��Ԥ������
		{
			ispre = true;
			cout << endl << "Ԥ���֡��Ƶ������" << endl;
		}
		if (c == 'q')//����q�˳�����Ԥ��
		{
			cout << "ֹͣԤ������" << endl;
			break;
		}
		if (ispre)
		{
			ispre = false;//����Ԥ���־λΪ�٣�ȷ��ÿ��ֻԤ��һ������
			frame.copyTo(temp);//���Ƶ�ǰ��Ƶ֡
			face = facedect(temp);//�ڵ�ǰ֡�м������
			if (face.empty())//û�м�⵽�������������
			{
				cout << endl << "û�м�⵽������" << endl;
				destroyWindow("����");
			}
			else//����������������²���
			{
				resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��
				cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
				toGrayscale(face);//��һ��				
				imshow("����", face);//����⵽��������ʾ����
				predictedLabel = model->predict(face);
				cout << "��⵽��" << predictedLabel << endl;

				// ��ʱ������Ҫ��û�����ģ���ڲ������ݣ�������cv::FaceRecognizer��ȴû�а취���
				// ����cv::FaceRecognizer����cv::Algorithm��������������Դ�cv::Algorithm�л�ȡ���ݡ�
				// ���ȣ���û������ѵ��ģ�͵�����£�����FaceRecognizer����ֵΪ0.0���⽫��ģ����������Ч��
				//model->set("threshold", 0.0);//û������ʲô��
				// ����ģ�͵���ֵΪ0.0�����ڲ���������֮����һ�����룬����Ԥ�⽫�᷵��-1
				//predictedLabel = model->predict(face);
				//cout << "Ԥ������� = " << predictedLabel << endl;
				// ��������λ��������ģ�͵�����ֵ��
				Mat eigenvalues = model->getMat("eigenvalues");
				// ͬ�������ǿ��Զ�ȡ���������������������
				Mat W = model->getMat("eigenvectors");
				// ��ʾǰ10��������:
				for (int i = 0; i < min(10, W.cols); i++)
				{
					string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
					cout << msg << endl;
					// ȡ���������� #i
					Mat ev = W.col(i).clone();
					// �仯Ϊԭ����ͼƬ��С���ҹ�һ����[0...255]������ʾ
					Mat grayscale = toGrayscale(ev.reshape(1, 200));//Mat grayscale = toGrayscale(ev.reshape(1, height));
					// ��ʾͼƬ�������ò�ɫͼƬ��ø��õ�Ч����
					Mat cgrayscale;
					applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
					imshow(format("%d_.jpg", i), cgrayscale);
					imshow(format("%d.jpg", i), grayscale);
					imwrite(format("%d_.jpg", i), cgrayscale);
					imwrite(format("%d.jpg", i), grayscale);
				}
			}			
		}
	}
}




//opencv�Դ�������
//#include "opencv2/core/core.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
//
//#include <iostream>
//#include <fstream>
//#include <sstream>
//
//using namespace cv;
//using namespace std;
//
////ֻ����Ϊ�Ҷ�ͼ�����򱨴�Ȼ�󽫻Ҷ�ͼ��һ��
//static Mat toGrayscale(InputArray _src)
//{
//	Mat src = _src.getMat();
//	// ֻ����Ϊ��ͨ����
//	if (src.channels() != 1) 
//	{
//		CV_Error(CV_StsBadArg, "ֻ֧�ֵ�ͨ���ľ���");
//	}
//	// ���������ع�һ�����ͼƬ
//	Mat dst;
//	cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
//	return dst;
//}
//
////��ȡCSV�ļ�
//static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') 
//{
//	std::ifstream file(filename.c_str(), ifstream::in);
//	if (!file)
//	{
//		string error_message = "û�и�������Ч�ļ��������������ļ�����";
//		CV_Error(CV_StsBadArg, error_message);
//	}
//	string line, path, classlabel;
//	while (getline(file, line))
//	{
//		stringstream liness(line);
//		getline(liness, path, separator);
//		getline(liness, classlabel);
//		if (!path.empty() && !classlabel.empty())
//		{
//			images.push_back(imread(path, 0));
//			labels.push_back(atoi(classlabel.c_str()));
//		}
//	}
//}
//
//int main(int argc, const char *argv[]) 
//{
//	// Check for valid command line arguments, print usage
//	// if no arguments were given.
//	//if (argc != 2) 
//	//{
//	//	cout << "usage: " << argv[0] << " <csv.ext>" << endl;
//	//	exit(1);
//	//}
//	// ����CSV�ļ����ļ���
//	string fn_csv = string("D:/face/kyl/cpp/PCA/PCA/yalefaces/face.csv");
//	// ����ͼƬ������������Ӧ��ŵ�����
//	vector<Mat> images;
//	vector<int> labels;
//	// ��ȡ���ݣ����û����Ч���ļ�����ʧ��
//	try 
//	{
//		read_csv(fn_csv, images, labels);
//	}
//	catch (cv::Exception& e)
//	{
//		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
//		// nothing more we can do
//		exit(1);
//	}
//	// ���û���㹻��ͼƬ���˳�
//	if (images.size() <= 1) 
//	{
//		string error_message = "��demo������Ҫ2��ͼƬ������ݼ�����㹻���ͼƬ��";
//		CV_Error(CV_StsError, error_message);
//	}
//	// �õ���һ��ͼƬ�ĸ߶ȣ���֮��Ĵ�����������Ҫ��ͼƬ��Ϊԭ���Ĵ�С
//	int height = images[0].rows;
//	// �������ļ��д���ֻ�Ǵ����ݼ��ж�ȡ���һ��ͼƬ�������������е�������������Ϊ��ʹѵ��������Լ����ص� 
//	// This is done, so that the training data (which we learn the
//	// cv::FaceRecognizer on) and the test data we test
//	// the model with, do not overlap.
//	Mat testSample = images[images.size() - 1];
//	int testLabel = labels[labels.size() - 1];
//	images.pop_back();
//	labels.pop_back();
//	// ���µļ��д�������������ʶ������������и�����CSV�ļ���������ͼƬ�ͱ��ѵ���ó�
//	//������һ����ȫ��PCA�����ֻ��Ҫ10�����ɷ֣���ôֻ��Ҫ�������ģ�������޸ļ��ɣ�
//	//      cv::createEigenFaceRecognizer(10);
//	//
//	// �����Ҫ����һ�����Ŷ���ֵ������ʶ���������µ�ģ������������
//	//      cv::createEigenFaceRecognizer(10, 123.0);
//	//
//	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
//	model->train(images, labels);
//	// һ�¼���Ԥ��һ�Ų���ͼƬ�ı��
//	int predictedLabel = model->predict(testSample);
//	//
//	// ���һ��Ԥ������Ŷ����������ģ������������:
//	//      int predictedLabel = -1;
//	//      double confidence = 0.0;
//	//      model->predict(testSample, predictedLabel, confidence);
//	//
//	string result_message = format("Ԥ������� = %d / ʵ�ʵ����� = %d.", predictedLabel, testLabel);
//	cout << result_message << endl;
//	// ��ʱ������Ҫ��û�����ģ���ڲ������ݣ�������cv::FaceRecognizer��ȴû�а취���
//	// ����cv::FaceRecognizer����cv::Algorithm��������������Դ�cv::Algorithm�л�ȡ���ݡ�
//	// ���ȣ���û������ѵ��ģ�͵�����£�����FaceRecognizer����ֵΪ0.0���⽫��ģ����������Ч��
//	model->set("threshold", 0.0);
//	// ����ģ�͵���ֵΪ0.0�����ڲ���������֮����һ�����룬����Ԥ�⽫�᷵��-1
//	predictedLabel = model->predict(testSample);
//	cout << "Ԥ������� = " << predictedLabel << endl;
//	// ��������λ��������ģ�͵�����ֵ��
//	Mat eigenvalues = model->getMat("eigenvalues");
//	// ͬ�������ǿ��Զ�ȡ���������������������
//	Mat W = model->getMat("eigenvectors");
//	// ��ʾǰ10��������:
//	for (int i = 0; i < min(10, W.cols); i++)
//	{
//		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
//		cout << msg << endl;
//		// ȡ���������� #i
//		Mat ev = W.col(i).clone();
//		// �仯Ϊԭ����ͼƬ��С���ҹ�һ����[0...255]������ʾ
//		Mat grayscale = toGrayscale(ev.reshape(1, height));
//		// ��ʾͼƬ�������ò�ɫͼƬ��ø��õ�Ч����
//		Mat cgrayscale;
//		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
//		imshow(format("%d", i), cgrayscale);
//	}
//	waitKey(0);
//
//	return 0;
//}