#pragma once
#include <iostream>

#include <opencv2/opencv.hpp>

#include "hog_feature.h"

using namespace std;
using namespace cv;

struct Watermark
{
	string imgFile;
	Rect box;
	Mat img;
	Watermark(Mat &src, Rect rin = Rect())
	{
		img = src;
		box = rin;
	}
};

class WatermarkProcess
{
	double roi_h_r;
	double roi_w_r;
	Size detect_size;

	double h_ratio_min;
	double h_ratio_max;

	bool outputRejectLevels;

public:
	CascadeClassifier watermark_cascade;
	CvSVM svm;
	HOG_feature feature;
	CvBoost boost;
	CvBoost boost_ii;

	WatermarkProcess(
		Size _detect_size = Size(42,16),
		double _h_ratio_min = 0,
		double _h_ratio_max = 1, 
		bool _outputRejectLevels = true,
		double roi_h_r=0,
	    double roi_w_r=0);

	Rect detect(const Mat& image,
		CV_OUT vector<Rect>& objects,
		vector<int>& rejectLevels,
		vector<double>& levelWeights,
		double &score,
		double scaleFactor = 1.1,
		int minNeighbors = 3);

	void wipeWatermark(Mat &img, Rect &posi);
	void showResult(Mat &img, const vector<Rect> &strs);
	void rectScale(Rect &box, double r);
};
