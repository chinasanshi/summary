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

// 保存图片的容器和相应标号的容器
vector<Mat> faces;
vector<int> labels;

static Mat toGrayscale(Mat src);
Mat facedect(Mat image);
void takphoto();
bool transfer(string fileName);
void predect(Ptr<FaceRecognizer> model);

int main(int argc, const char *argv[])
{
	//takphoto();//采集人脸样本
	transfer("D:\\face\\kyl\\cpp\\PCA\\PCA\\*.jpg");//遍历该路径下的所有符合命名规则的.jpg图片
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();//定义PCA模型
	model->train(faces, labels);//训练模型

	predect(model);//打开摄像头，预测人脸
	
	waitKey(0);

	return 0;
}


Mat facedect(Mat image)//检测一张图片里的人脸
{
	char* cascade_name = "C:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml";
	CascadeClassifier frontalface_cascade;
	if (!frontalface_cascade.load(cascade_name))//判断Haar特征加载是否成功
	{
		printf("无法加载级联分类器文件！\n");
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
//只允许为灰度图，否则报错，然后将灰度图归一化
static Mat toGrayscale(Mat src)
{
	// 只允许为单通道的
	if (src.channels() != 1)
	{
		CV_Error(CV_StsBadArg, "只支持单通道的矩阵！");
	}
	// 创建并返回归一化后的图片
	Mat dst;
	cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}
  
//如果按键‘b’，开始采集人脸，如果按键‘f’，此人人脸采集结束，按键‘q’，退出采集人脸函数，表示所有人脸样本采集完毕
void takphoto()
{
	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << "打开摄像头失败！" << endl;
	}

	bool beginface = false;//是否开始采集人脸的标志位

	Mat frame;//保存视频帧
	Mat temp;//采集人脸时临时复制当前视频帧
	Mat face;//保存采集到的人脸
	//Mat not;//保存按位取反后的视频图像
	namedWindow("img");
	//vector<Mat> faces;
	//vector<int> labels;
	int label = 0;//当前人脸的标号，每采集一次自加1
	int faceNo = 0;//保存当前采集人脸的个数
	int AllfaceNo = 0;//采集人脸的总数
	long frameNo = 0;//视频的帧数
	long NowframNo = 0;//保存开始采集人脸时的视频帧数

	string name = "f";

	while (capture.read(frame))
	{
		frameNo++;//帧数加1
		cout << "当前帧数为" << frameNo << "; "; //输出当前帧数
		//bitwise_not(frame,not);//将frame按位取反得到取反后的图像
		//imshow("not", not);//显示按位取反后的图像
		imshow("img", frame);//显示视频帧
		char c = waitKey(33);//控制帧率
		if (c == 'b')//如果按键‘b’，开始采集人脸
		{
			beginface = true;//将标志位设为真
			label++;//开始一次人脸采集label自加1，用来区别不同人脸
			name = "f";//每次开始采集人脸都重新给名字幅值
			stringstream ss;//stringstream可以吞下不同的类型，根据s1的类型，然后吐出不同的类型
			ss << label;
			string s1 = ss.str();//变为字符串
			name += s1;//name变为n加上标号，如n1,n2……	
			name += "_";//name变为n1_,n2_……	
			NowframNo = frameNo;//保存当前的帧数，用以判断当前帧开始后的每30帧
			cout << endl << endl << "开始采集人脸，此人标号为" << label << endl << endl;
		}
		//满足开始检测的条件，从当前时刻开始每30帧采集一次人脸
		if (((beginface) && (frameNo - NowframNo - 30) % 30 == 0))
		{
			frame.copyTo(temp);//复制当前视频帧
			face = facedect(temp);//在当前帧中检测人脸
			if (face.empty())//没有检测到人脸则显示错误
			{
				cout<< endl << endl << "没有检测到人脸，请对准摄像头！" << endl << endl;
				//break;//不可以用break，否则会退出样本采集
			}
			else//若有检测到人脸则进行如下操作
			{
				resize(face, face, Size(200, 200));//将人脸都变为200*200大小的图像
				cvtColor(face, face, CV_BGR2GRAY);//转换为灰度图
				toGrayscale(face);//归一化
				faces.push_back(face);//将人脸压栈
				labels.push_back(label);//将标号压栈
				faceNo++;//每进入一次循环人脸数自加1，用来记录此次采集的人脸总数
				AllfaceNo++;//保存总共采集到的人脸的个数
				stringstream ss;//stringstream可以吞下不同的类型，根据s1的类型，然后吐出不同的类型
				ss << faceNo;
				string s1 = ss.str();//变为字符串
				name += s1;//name变为n加上标号，如n1,n2……	
				name += ".jpg";
				namedWindow("face");
				imshow("face", face);//显示此次采集到的人脸			
				imwrite(name, face);
				cout <<endl << endl << "检测到并保存人脸图片" << name << endl;
				name.pop_back();
				name.pop_back();
				name.pop_back();
				name.pop_back();//以上4次是弹出".jpg"
				name.pop_back();//弹出标号
				if (faceNo > 9)//如果样本个数为两位数需要多弹出一次，如果此人样本数大于99个则保存的头像的名字会出错
				{
					name.pop_back();
				}
				cout << "目前采集到此人脸的个数为" << faceNo << endl << endl;
			}
		}
		if (c == 'f')//如果按键‘f’，此人人脸采集结束
		{
			beginface = false;//阻值下一次采集
			faceNo -= 1;
			AllfaceNo -= 1;
			faces.pop_back();
			labels.pop_back();//弹出最后一次采集的人脸，因为效果不好
			cout << "采集此人脸完毕，总共采集此人样本数为" << faceNo << endl;
			faceNo = 0;//将人脸数清零，为下一位人脸采集做准备
		}
		if (c == 'q')//按键‘q’，退出采集人脸函数，表示所有人脸样本采集完毕
		{
			cout << "人脸样本采集完毕！" << endl;
			cout << "总共采集人脸样本个数为"<< AllfaceNo << endl;
			break;
		}
	}
	destroyWindow("img");//关闭视频窗口
	destroyWindow("face");//关闭face窗口
}


//结构体说明
//struct _finddata_t
//{
//	unsigned attrib;     //文件属性
//	time_t time_create;  //文件创建时间
//	time_t time_access;  //文件上一次访问时间
//	time_t time_write;   //文件上一次修改时间
//	_fsize_t size;  //文件字节数
//	char name[_MAX_FNAME]; //文件名
//};
////按FileName命名规则匹配当前目录第一个文件
//_findfirst(_In_ const char * FileName, _Out_ struct _finddata64i32_t * _FindData);
////按FileName命名规则匹配当前目录下一个文件
//_findnext(_In_ intptr_t _FindHandle, _Out_ struct _finddata64i32_t * _FindData);
////关闭_findfirst返回的文件句柄
//_findclose(_In_ intptr_t _FindHandle);

bool transfer(string fileName)//filename允许有通配符，'？'代表一个字符，'*'代表0到任意字符
{
	int jpgNum = 0;//记录总共找到的图片的个数
	char labelchar[10]="\0";//保存从图片名字里面提取到的标号
	int label = 0;
	_finddata_t fileInfo;//定义结构体
	long handle = _findfirst(fileName.c_str(), &fileInfo);//寻找第一个图片文件

	if (handle == -1L)//如果没有找到文件则句柄返回-1L
	{
		cerr << "未找到图片文件！" << endl;
		return false;
	}

	do
	{	
		if (fileInfo.name[0] != 'f')//如果图片名称不是以f开头则不符合条件，进入下一次查找
		{
			continue;//不可用break，否则会直接退出do-while循环
		}
		for (int i = 0; i < strlen(fileInfo.name); i++)
		{
			if (fileInfo.name[i + 1] == '_')//遇到下划线则退出for循环
			{
				labelchar[i] = '\0';
				break;
			}
			labelchar[i] = fileInfo.name[i + 1];//图片名字的第一个字母为f，不用保留
		}
		label = atoi(labelchar);//将char型标号转换成int型
		labels.push_back(label);//将标号压栈传入标号容器
		faces.push_back(imread(fileInfo.name,0));//将找到的图片以灰度形式压栈出入faces，否则不会有结果
		jpgNum++;//图片数加1
		cout << fileInfo.name << endl;//输出找到的图片的名字
	} while (_findnext(handle, &fileInfo) == 0);//寻找下一张图片
	cout << " 有效 .jpg 图片文件的个数为:  " << jpgNum << endl;//输出找到的图片的个数

	return true;
}

//按键p则预测人脸，按键q退出人脸预测
void predect(Ptr<FaceRecognizer> model)
{
	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << "打开摄像头失败！" << endl;
	}
	Mat frame;//保存视频帧
	Mat temp;
	Mat face;
	namedWindow("人脸");
	bool ispre = false;
	int predictedLabel;
	while (capture.read(frame))
	{
		char c = waitKey(33);//控制帧率
		imshow("predect", frame);//显示视频
		if (c == 'p')//按键p则预测人脸
		{
			ispre = true;
			cout << endl << "预测此帧视频的人脸" << endl;
		}
		if (c == 'q')//按键q退出人脸预测
		{
			cout << "停止预测人脸" << endl;
			break;
		}
		if (ispre)
		{
			ispre = false;//人脸预测标志位为假，确保每次只预测一次人脸
			frame.copyTo(temp);//复制当前视频帧
			face = facedect(temp);//在当前帧中检测人脸
			if (face.empty())//没有检测到人脸则输出错误
			{
				cout << endl << "没有检测到人脸！" << endl;
				destroyWindow("人脸");
			}
			else//若有人脸则进行如下操作
			{
				resize(face, face, Size(200, 200));//将人脸都变为200*200大小的图像
				cvtColor(face, face, CV_BGR2GRAY);//转换为灰度图
				toGrayscale(face);//归一化				
				imshow("人脸", face);//将检测到的人脸显示出来
				predictedLabel = model->predict(face);
				cout << "检测到是" << predictedLabel << endl;

				// 有时候你想要获得或设置模型内部的数据，但是在cv::FaceRecognizer中却没有办法获得
				// 由于cv::FaceRecognizer是由cv::Algorithm派生而来，你可以从cv::Algorithm中获取数据。
				// 首先，在没有重新训练模型的情况下，设置FaceRecognizer的阈值为0.0。这将对模型评估很有效。
				//model->set("threshold", 0.0);//没看懂有什么用
				// 现在模型的阈值为0.0。由于不可能在它之下有一个距离，现在预测将会返回-1
				//predictedLabel = model->predict(face);
				//cout << "预测的类是 = " << predictedLabel << endl;
				// 下面是如何获得特征脸模型的特征值：
				Mat eigenvalues = model->getMat("eigenvalues");
				// 同样的我们可以读取特征脸来获得特征向量：
				Mat W = model->getMat("eigenvectors");
				// 显示前10个特征脸:
				for (int i = 0; i < min(10, W.cols); i++)
				{
					string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
					cout << msg << endl;
					// 取得特征向量 #i
					Mat ev = W.col(i).clone();
					// 变化为原来的图片大小并且归一化到[0...255]用来显示
					Mat grayscale = toGrayscale(ev.reshape(1, 200));//Mat grayscale = toGrayscale(ev.reshape(1, height));
					// 显示图片并且运用彩色图片获得更好的效果。
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




//opencv自带的例程
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
////只允许为灰度图，否则报错，然后将灰度图归一化
//static Mat toGrayscale(InputArray _src)
//{
//	Mat src = _src.getMat();
//	// 只允许为单通道的
//	if (src.channels() != 1) 
//	{
//		CV_Error(CV_StsBadArg, "只支持单通道的矩阵！");
//	}
//	// 创建并返回归一化后的图片
//	Mat dst;
//	cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
//	return dst;
//}
//
////读取CSV文件
//static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') 
//{
//	std::ifstream file(filename.c_str(), ifstream::in);
//	if (!file)
//	{
//		string error_message = "没有给出的有效文件，请检查所给的文件名。";
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
//	// 给出CSV文件的文件名
//	string fn_csv = string("D:/face/kyl/cpp/PCA/PCA/yalefaces/face.csv");
//	// 保存图片的容器，和相应标号的容器
//	vector<Mat> images;
//	vector<int> labels;
//	// 读取数据，如果没有有效的文件名会失败
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
//	// 如果没有足够的图片会退出
//	if (images.size() <= 1) 
//	{
//		string error_message = "该demo至少需要2张图片请给数据集添加足够多的图片！";
//		CV_Error(CV_StsError, error_message);
//	}
//	// 得到第一张图片的高度，在之后的代码中我们需要将图片变为原来的大小
//	int height = images[0].rows;
//	// 接下来的几行代码只是从数据集中读取最后一张图片并把它从容器中弹出，这样做是为了使训练集与测试集不重叠 
//	// This is done, so that the training data (which we learn the
//	// cv::FaceRecognizer on) and the test data we test
//	// the model with, do not overlap.
//	Mat testSample = images[images.size() - 1];
//	int testLabel = labels[labels.size() - 1];
//	images.pop_back();
//	labels.pop_back();
//	// 以下的几行创建了用于人脸识别的特征脸，有给定的CSV文件所给出的图片和标号训练得出
//	//这里是一个完全的PCA，如果只需要10个主成分，那么只需要像下面的模型那样修改即可：
//	//      cv::createEigenFaceRecognizer(10);
//	//
//	// 如果想要创建一个置信度阈值的人脸识别器像如下的模型那样做即可
//	//      cv::createEigenFaceRecognizer(10, 123.0);
//	//
//	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
//	model->train(images, labels);
//	// 一下几行预测一张测试图片的标号
//	int predictedLabel = model->predict(testSample);
//	//
//	// 获得一个预测的置信度需像下面的模型那样做即可:
//	//      int predictedLabel = -1;
//	//      double confidence = 0.0;
//	//      model->predict(testSample, predictedLabel, confidence);
//	//
//	string result_message = format("预测的类是 = %d / 实际的类是 = %d.", predictedLabel, testLabel);
//	cout << result_message << endl;
//	// 有时候你想要获得或设置模型内部的数据，但是在cv::FaceRecognizer中却没有办法获得
//	// 由于cv::FaceRecognizer是由cv::Algorithm派生而来，你可以从cv::Algorithm中获取数据。
//	// 首先，在没有重新训练模型的情况下，设置FaceRecognizer的阈值为0.0。这将对模型评估很有效。
//	model->set("threshold", 0.0);
//	// 现在模型的阈值为0.0。由于不可能在它之下有一个距离，现在预测将会返回-1
//	predictedLabel = model->predict(testSample);
//	cout << "预测的类是 = " << predictedLabel << endl;
//	// 下面是如何获得特征脸模型的特征值：
//	Mat eigenvalues = model->getMat("eigenvalues");
//	// 同样的我们可以读取特征脸来获得特征向量：
//	Mat W = model->getMat("eigenvectors");
//	// 显示前10个特征脸:
//	for (int i = 0; i < min(10, W.cols); i++)
//	{
//		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
//		cout << msg << endl;
//		// 取得特征向量 #i
//		Mat ev = W.col(i).clone();
//		// 变化为原来的图片大小并且归一化到[0...255]用来显示
//		Mat grayscale = toGrayscale(ev.reshape(1, height));
//		// 显示图片并且运用彩色图片获得更好的效果。
//		Mat cgrayscale;
//		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
//		imshow(format("%d", i), cgrayscale);
//	}
//	waitKey(0);
//
//	return 0;
//}