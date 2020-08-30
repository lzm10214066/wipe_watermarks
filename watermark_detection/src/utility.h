#pragma once
#include <iostream>

#include <opencv2/opencv.hpp>

#include <set>

#include "watermark.h"

using namespace std;
using namespace cv;

string getFileName(string str);
void getRangeFromTxt(string tempObject, int *xmin, int *xmax, int *ymin, int *ymax,const Mat &img);
void processData(vector<string> &rects_str, string &str_box);
void getWatermarkData(string tempStrData, Watermark &watermark,const Mat &img);
double iou(const Rect &r1, const Rect &r2);
double overlapArea(const cv::Rect &a, const cv::Rect &b);
void normSizeByOneSide(Mat &recImage, double length);
void prepareSamples(const Mat &src, const Rect &bbox, const Size &de_window, vector<Rect> &sampels);
void rectScale(Rect &box, double r);
void filter_hard(vector<Rect> &hard, Mat &img, WatermarkProcess &filter);
