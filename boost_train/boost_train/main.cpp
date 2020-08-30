#include <opencv2/core/core.hpp>
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\ml\ml.hpp"

#include <fstream>
using namespace cv;
using namespace std;

int main(int ac, char** av)
{
	string filename = av[1];
	cout << "lables+feature file: " << filename << endl;
//////////////////////////////////////////////////////////////////
	FileStorage fs;
	fs.open(filename, FileStorage::READ);
	if (!fs.isOpened())
	{
		cerr << "Failed to open the feature file " << filename << endl;
		return 1;
	}
	else
	{
		cout << "load the feature file successfully!" << endl;
	}
	///////////////////////////////////////////////////////////////////

	int imageNum = 0, featureDim=0;
	fs["imageNum"] >> imageNum;
	fs["featureDim"] >> featureDim;

	FileNode lable_feature = fs["lable_feature"];
	float *labels = new float[imageNum];
	vector <vector <float>   >   trainingData(imageNum, vector <float>(featureDim));
	//int   **trainingData = new   int*[];

	for (int i = 0; i < imageNum; ++i)              //读取，先lable，后跟着特征，重复。
	{
		lable_feature[2 * i] >> labels[i];
		lable_feature[2 * i + 1] >> trainingData[i];
	}

	Mat labelsMat(imageNum, 1, CV_32FC1, labels);    //调用构造函数转为Mat格式
	Mat trainingDataMat(imageNum, featureDim, CV_32FC1);
	for (int i = 0; i < imageNum; ++i)                //不知道如何用二维vector进行构造函数，姑且每个进行赋值吧
	{
		for (int j = 0; j < featureDim; ++j)
		{
			trainingDataMat.at<float>(i, j) = trainingData[i][j];
		}
	}
/////////////////////////////////////*boost train*//////////////////////////////////////////////////
	double t = getTickCount();

	string boost_filename = filename.erase(filename.find('.')) + "-boost.xml";
	CvBoost  boost;
	Mat var_types(1, trainingDataMat.cols + 1, CV_8UC1, Scalar(CV_VAR_ORDERED));
	var_types.at<uchar>(trainingDataMat.cols) = CV_VAR_CATEGORICAL;

	int weak_count = atoi(av[2]);
	CvBoostParams  params(CvBoost::DISCRETE, // boost_type
		weak_count, // weak_count
		0.95, // weight_trim_rate
		2, // max_depth
		false, //use_surrogates
		0 // priors
		);

	cout << "boost training..............." << endl;

	boost.train(trainingDataMat, CV_ROW_SAMPLE, labelsMat, Mat(), Mat(), var_types, Mat(), params);

	boost.save(boost_filename.c_str());
	cout << "boost file: " << boost_filename << " saved" << endl;

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Times passed in seconds: " << t << "s" << endl;

	waitKey(0);
	return 0;
}