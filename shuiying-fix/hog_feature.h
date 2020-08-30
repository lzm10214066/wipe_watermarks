#pragma once
#include <iostream>

#include <opencv2/opencv.hpp>

#include <set>

using namespace std;
using namespace cv;

struct HOG_feature
{
private:
	int nbins;
	Size blockSize;
	Size blockStride;
	Size NormSize;

	void calculateHOG(Mat &src, vector<float> &descriptors);
	void calculateColorHOG(Mat &src, vector<float> &descriptors);
	double hog_classify_one_svm(Mat &img, CvSVM &svm);
	double hog_color_classify_one_svm(Mat &img, CvSVM &svm);

public:
	HOG_feature(int _nbins = 8, 
		Size _blockSize = Size(8, 8), 
		Size _blockStride = Size(4, 4), 
		Size _NormSize = Size(48, 16),
		bool _color=false,
		double _s_th=0);

	bool color;
	double s_th;

	void hog_filter(Mat &img, vector<Rect> &strs, double th, CvSVM &svm,bool color);
	void hog_filter(Mat &img, vector<Rect> &strs, double th, CvSVM &svm, bool color,vector<double> &scores);

	double hog_classify_one_boost(Mat &img, CvBoost &boost);
	void hog_filter(Mat &img, vector<Rect> &strs, double th, CvBoost &boost, vector<double> &scores);

};

