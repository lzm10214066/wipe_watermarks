#ifndef _IMAGEREAD_H
#define  _IMAGEREAD_H

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <direct.h>
#include<io.h>

using namespace std;

int readImageFile(string &imageFile, vector<string> &pathOfImage);

void rect2image(Mat &image, vector<Rect> recBag, string imagePath, string &folder, int type);
void rectsSaveAsImage(Mat &image, vector<Rect> recBag, string imagePath, string &folder);
void rectsSaveAsImage(Mat &image, vector<Rect> recBag, string imagePath, string &folder, Size normSize);
void getFiles(string path, vector<string>& files);


#endif