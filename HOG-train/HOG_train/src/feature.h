#pragma once
#include <iostream>

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\objdetect\objdetect.hpp"

#include <set>

using namespace std;
using namespace cv;

void calculateHOG(Mat &src, vector<float> &descriptors);
double hog_classify_one_svm(Mat &img, CvSVM &svm);
void calculateColorHOG(Mat &src, vector<float> &descriptors);
double hog_color_classify_one_svm(Mat &img, CvSVM &svm);
double hog_classify_one_boost(Mat &img, CvBoost &boost);


