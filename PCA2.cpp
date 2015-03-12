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
	takphoto();//采集人脸样本，保存成头像图片文件
	transfer("*.jpg");//遍历该路径下的所有符合命名规则的.jpg图片
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
	int label = 0;//当前人脸的标号，每采集一次自加1
	string labelcin;//定义一个接收标号的字符串
	string facenumcin;//定义一个接收输入人脸序号的字符串
	string facenum;//定义一个人脸序号的字符串，用以保存图片名称时使用
	int faceno = 0;//存储人脸的标号
	int facetotal = 0;//保存当前采集人脸的个数
	int Allfacetotal = 0;//采集人脸的总数
	long frameNo = 0;//视频的帧数
	long NowframNo = 0;//保存开始采集人脸时的视频帧数

	string name = "f";

	while (capture.read(frame))
	{
		frameNo++;//帧数加1
		//cout << "当前帧数为" << frameNo << "; "; //输出当前帧数
		//bitwise_not(frame,not);//将frame按位取反得到取反后的图像
		//imshow("not", not);//显示按位取反后的图像
		imshow("img", frame);//显示视频帧
		char c = waitKey(33);//控制帧率
		if (c == 'b')//如果按键‘b’，开始采集人脸
		{
			while (1)
			{
				cout << "请输入待采集人脸的标号：";				
				cin >> labelcin;//输入标号
				int labelnum = 0;//存储输入字符串中数字的个数
				for (unsigned int l = 0; l < labelcin.length(); l++)//循环判断输入的标号字符串是否是数字
				{
					if (!isdigit(labelcin[l]))//使用isdigit函数可以判断每一位是否是0-9的数字，如果输入的字符串有有任何一位不是数字则要重新输入
					{
						cout << endl << "请输入数字！" << endl;						
						break;//退出for循环
					}
					else
					{
						labelnum++;//如果判断此次循环的字符串中的是数字，labelnum加1
					}					
				}
				if (labelnum == labelcin.length())//如果数字的长度等于输入字符串的长度，表示输入的都是数字
				{
					stringstream ss1;//stringstream可以吞下不同的类型，根据要求的类型，然后吐出不同的类型
					ss1 << labelcin;
					ss1 >> label;//将输入的标号转换为int型，从而可以将标号压栈存储
					break;//退出while循环
				}
			}
			while (1)
			{
				cout << "请输入待采集人脸开始的序号：";
				cin >> facenumcin;//输入标号
				int facenlen = 0;//存储输入字符串中数字的个数
				for (unsigned int f = 0; f < facenumcin.length(); f++)//循环判断输入的标号字符串是否是数字
				{
					if (!isdigit(facenumcin[f]))//使用isdigit函数可以判断每一位是否是0-9的数字，如果输入的字符串有有任何一位不是数字则要重新输入
					{
						cout << endl << "请输入数字！" << endl;
						break;//退出for循环
					}
					else
					{
						facenlen++;//如果判断此次循环的字符串中的是数字，labelnum加1
					}
				}
				if (facenlen == facenumcin.length())//如果数字的长度等于输入字符串的长度，表示输入的都是数字
				{
					stringstream ss2;//stringstream可以吞下不同的类型，根据要求的类型，然后吐出不同的类型
					ss2 << facenumcin;
					ss2 >> faceno;//将输入的序号转换为int型，从而可以自加1，以区别人脸的名称
					faceno -= 1;//先将输入的值减去1，因为后面的循环里有加1，使得采集头像的序号从输入的序号开始
					break;//退出while循环
				}
			}
			//如果输入的标号和序号满足要求则进入如下操作
			beginface = true;//将标志位设为真
			name = "f";//每次开始采集人脸都重新给名字幅值
			name += labelcin;//name变为n加上标号，如n1,n2……	
			name += "_";//name变为n1_,n2_……	
			NowframNo = frameNo;//保存当前的帧数，用以判断当前帧开始后的每30帧
			cout << endl << endl << "开始采集人脸，此人标号为" << label << endl << endl;			
		}
		//满足开始检测的条件，从当前时刻开始每30帧采集一次人脸
		if (((beginface) && (frameNo - NowframNo) % 30 == 0))
		{
			frame.copyTo(temp);//复制当前视频帧
			face = facedect(temp);//在当前帧中检测人脸
			if (face.empty())//没有检测到人脸则显示错误
			{
				cout << endl << endl << "没有检测到人脸，请对准摄像头！" << endl << endl;
				//break;//不可以用break，否则会退出样本采集
			}
			else//若有检测到人脸则进行如下操作
			{
				resize(face, face, Size(200, 200));//将人脸都变为200*200大小的图像
				cvtColor(face, face, CV_BGR2GRAY);//转换为灰度图
				toGrayscale(face);//归一化
				//faces.push_back(face);//将人脸压栈
				//labels.push_back(label);//将标号压栈
				facetotal++;//每进入一次循环人脸数自加1，用来记录此次采集的人脸总数
				Allfacetotal++;//保存总共采集到的人脸的个数
				faceno++;//将输入的序号自加1，以区别人脸的名称
				stringstream ss3;//stringstream可以吞下不同的类型，根据要求的类型，然后吐出不同的类型
				ss3 << faceno;//吞下faceno，转换成string类型（facenum）用以保存图片名称
				ss3>> facenum;//将输入的序号转换为int型，从而可以自加1，以区别人脸的名称
				name += facenum;//name变为n加上标号，如n1_1,n1_2……	
				name += ".jpg";
				namedWindow("face");
				imshow("face", face);//显示此次采集到的人脸			
				imwrite(name, face);
				cout << endl << endl << "检测到并保存人脸图片" << name << endl;
				name.pop_back();
				name.pop_back();
				name.pop_back();
				name.pop_back();//以上4次是弹出".jpg"
				name.pop_back();//弹出标号
				if (faceno > 9)//如果样本名称的序号为两位数需要多弹出一次
				{
					name.pop_back();
				}
				if (faceno > 99)//如果样本名称的序号为3位数还需要多弹出一次，如果此人样本名称的序号大于999个则保存的头像的名字会出错
				{
					name.pop_back();
				}
				cout << "目前采集到此人脸的个数为" << facetotal << endl << endl;
			}
		}
		if (c == 'f')//如果按键‘f’，此人人脸采集结束
		{
			beginface = false;//阻值下一次采集			
			cout << "采集此人脸完毕！此次总共采集此人样本数为" << facetotal << endl;
			facetotal = 0;//将人脸数清零，为下一位人脸采集做准备。此句需在上面的输出语句的后面，否则输出的总采集人脸数就被清零了
			//system("del name");
			//facetotal -= 1;
			//Allfacetotal -= 1;			
			//faces.pop_back();
			//labels.pop_back();//弹出最后一次采集的人脸，因为效果不好
			//cout << "采集此人脸完毕！舍弃最后一次采集的头像，总共采集此人样本数为" << facetotal << endl;
		}
		if (c == 'q')//按键‘q’，退出采集人脸函数，表示所有人脸样本采集完毕
		{
			cout << "人脸样本采集完毕！" << endl;
			cout << "总共采集人脸样本个数为" << Allfacetotal << endl;
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
	char labelchar[10] = "\0";//保存从图片名字里面提取到的标号
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
		for (unsigned int i = 0; i < strlen(fileInfo.name); i++)
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
		faces.push_back(imread(fileInfo.name, 0));//将找到的图片以灰度形式压栈出入faces，否则不会有结果
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

