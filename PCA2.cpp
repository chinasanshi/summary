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
#include<conio.h>   
//#include <libxml/parser.h>  
//#include <libxml/tree.h>


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
	takphoto();//�ɼ����������������ͷ��ͼƬ�ļ�
	transfer("*.jpg");//������·���µ����з������������.jpgͼƬ
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
	int label = 0;//��ǰ�����ı�ţ�ÿ�ɼ�һ���Լ�1
	string labelcin;//����һ�����ձ�ŵ��ַ���
	string facenumcin;//����һ����������������ŵ��ַ���
	string facenum;//����һ��������ŵ��ַ��������Ա���ͼƬ����ʱʹ��
	int faceno = 0;//�洢�����ı��
	int facetotal = 0;//���浱ǰ�ɼ������ĸ���
	int Allfacetotal = 0;//�ɼ�����������
	long frameNo = 0;//��Ƶ��֡��
	long NowframNo = 0;//���濪ʼ�ɼ�����ʱ����Ƶ֡��

	string name = "f";

	while (capture.read(frame))
	{
		frameNo++;//֡����1
		//cout << "��ǰ֡��Ϊ" << frameNo << "; "; //�����ǰ֡��
		//bitwise_not(frame,not);//��frame��λȡ���õ�ȡ�����ͼ��
		//imshow("not", not);//��ʾ��λȡ�����ͼ��
		imshow("img", frame);//��ʾ��Ƶ֡
		char c = waitKey(33);//����֡��
		if (c == 'b')//���������b������ʼ�ɼ�����
		{
			while (1)
			{
				cout << "��������ɼ������ı�ţ�";				
				cin >> labelcin;//������
				int labelnum = 0;//�洢�����ַ��������ֵĸ���
				for (unsigned int l = 0; l < labelcin.length(); l++)//ѭ���ж�����ı���ַ����Ƿ�������
				{
					if (!isdigit(labelcin[l]))//ʹ��isdigit���������ж�ÿһλ�Ƿ���0-9�����֣����������ַ��������κ�һλ����������Ҫ��������
					{
						cout << endl << "���������֣�" << endl;						
						break;//�˳�forѭ��
					}
					else
					{
						labelnum++;//����жϴ˴�ѭ�����ַ����е������֣�labelnum��1
					}					
				}
				if (labelnum == labelcin.length())//������ֵĳ��ȵ��������ַ����ĳ��ȣ���ʾ����Ķ�������
				{
					stringstream ss1;//stringstream�������²�ͬ�����ͣ�����Ҫ������ͣ�Ȼ���³���ͬ������
					ss1 << labelcin;
					ss1 >> label;//������ı��ת��Ϊint�ͣ��Ӷ����Խ����ѹջ�洢
					break;//�˳�whileѭ��
				}
			}
			while (1)
			{
				cout << "��������ɼ�������ʼ����ţ�";
				cin >> facenumcin;//������
				int facenlen = 0;//�洢�����ַ��������ֵĸ���
				for (unsigned int f = 0; f < facenumcin.length(); f++)//ѭ���ж�����ı���ַ����Ƿ�������
				{
					if (!isdigit(facenumcin[f]))//ʹ��isdigit���������ж�ÿһλ�Ƿ���0-9�����֣����������ַ��������κ�һλ����������Ҫ��������
					{
						cout << endl << "���������֣�" << endl;
						break;//�˳�forѭ��
					}
					else
					{
						facenlen++;//����жϴ˴�ѭ�����ַ����е������֣�labelnum��1
					}
				}
				if (facenlen == facenumcin.length())//������ֵĳ��ȵ��������ַ����ĳ��ȣ���ʾ����Ķ�������
				{
					stringstream ss2;//stringstream�������²�ͬ�����ͣ�����Ҫ������ͣ�Ȼ���³���ͬ������
					ss2 << facenumcin;
					ss2 >> faceno;//����������ת��Ϊint�ͣ��Ӷ������Լ�1������������������
					faceno -= 1;//�Ƚ������ֵ��ȥ1����Ϊ�����ѭ�����м�1��ʹ�òɼ�ͷ�����Ŵ��������ſ�ʼ
					break;//�˳�whileѭ��
				}
			}
			//�������ı�ź��������Ҫ����������²���
			beginface = true;//����־λ��Ϊ��
			name = "f";//ÿ�ο�ʼ�ɼ����������¸����ַ�ֵ
			name += labelcin;//name��Ϊn���ϱ�ţ���n1,n2����	
			name += "_";//name��Ϊn1_,n2_����	
			NowframNo = frameNo;//���浱ǰ��֡���������жϵ�ǰ֡��ʼ���ÿ30֡
			cout << endl << endl << "��ʼ�ɼ����������˱��Ϊ" << label << endl << endl;			
		}
		//���㿪ʼ�����������ӵ�ǰʱ�̿�ʼÿ30֡�ɼ�һ������
		if (((beginface) && (frameNo - NowframNo) % 30 == 0))
		{
			frame.copyTo(temp);//���Ƶ�ǰ��Ƶ֡
			face = facedect(temp);//�ڵ�ǰ֡�м������
			if (face.empty())//û�м�⵽��������ʾ����
			{
				cout << endl << endl << "û�м�⵽���������׼����ͷ��" << endl << endl;
				//break;//��������break��������˳������ɼ�
			}
			else//���м�⵽������������²���
			{
				resize(face, face, Size(200, 200));//����������Ϊ200*200��С��ͼ��
				cvtColor(face, face, CV_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
				toGrayscale(face);//��һ��
				//faces.push_back(face);//������ѹջ
				//labels.push_back(label);//�����ѹջ
				facetotal++;//ÿ����һ��ѭ���������Լ�1��������¼�˴βɼ�����������
				Allfacetotal++;//�����ܹ��ɼ����������ĸ���
				faceno++;//�����������Լ�1������������������
				stringstream ss3;//stringstream�������²�ͬ�����ͣ�����Ҫ������ͣ�Ȼ���³���ͬ������
				ss3 << faceno;//����faceno��ת����string���ͣ�facenum�����Ա���ͼƬ����
				ss3>> facenum;//����������ת��Ϊint�ͣ��Ӷ������Լ�1������������������
				name += facenum;//name��Ϊn���ϱ�ţ���n1_1,n1_2����	
				name += ".jpg";
				namedWindow("face");
				imshow("face", face);//��ʾ�˴βɼ���������			
				imwrite(name, face);
				cout << endl << endl << "��⵽����������ͼƬ" << name << endl;
				name.pop_back();
				name.pop_back();
				name.pop_back();
				name.pop_back();//����4���ǵ���".jpg"
				name.pop_back();//�������
				if (faceno > 9)//����������Ƶ����Ϊ��λ����Ҫ�൯��һ��
				{
					name.pop_back();
				}
				if (faceno > 99)//����������Ƶ����Ϊ3λ������Ҫ�൯��һ�Σ���������������Ƶ���Ŵ���999���򱣴��ͷ������ֻ����
				{
					name.pop_back();
				}
				cout << "Ŀǰ�ɼ����������ĸ���Ϊ" << facetotal << endl << endl;
			}
		}
		if (c == 'f')//���������f�������������ɼ�����
		{
			beginface = false;//��ֵ��һ�βɼ�			
			cout << "�ɼ���������ϣ��˴��ܹ��ɼ�����������Ϊ" << facetotal << endl;
			facetotal = 0;//�����������㣬Ϊ��һλ�����ɼ���׼�����˾����������������ĺ��棬����������ܲɼ��������ͱ�������
			//system("del name");
			//facetotal -= 1;
			//Allfacetotal -= 1;			
			//faces.pop_back();
			//labels.pop_back();//�������һ�βɼ�����������ΪЧ������
			//cout << "�ɼ���������ϣ��������һ�βɼ���ͷ���ܹ��ɼ�����������Ϊ" << facetotal << endl;
		}
		if (c == 'q')//������q�����˳��ɼ�������������ʾ�������������ɼ����
		{
			cout << "���������ɼ���ϣ�" << endl;
			cout << "�ܹ��ɼ�������������Ϊ" << Allfacetotal << endl;
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
	char labelchar[10] = "\0";//�����ͼƬ����������ȡ���ı��
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
		for (unsigned int i = 0; i < strlen(fileInfo.name); i++)
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
		faces.push_back(imread(fileInfo.name, 0));//���ҵ���ͼƬ�ԻҶ���ʽѹջ����faces�����򲻻��н��
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

