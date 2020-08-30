#include <stdio.h>
#include <stdlib.h>
#include <io.h>

#include <iostream>
#include <fstream>

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\photo\photo.hpp"

#include "../src/imageRead.h"
#include "../src/utility.h"
#include "../src/feature.h"

using namespace cv;
using namespace std;

int main(void)
{
	//////////////////////////////////////*read path.txt*///////////////////////////
	string pos_imageFile = "../img_path/pos_path.txt";
	vector<string> pos_image_path;
	int posCount = readImageFile(pos_imageFile, pos_image_path);

	string neg_imageFile = "../img_path/neg_stage_2_path.txt";
	vector<string> neg_image_path;
	int negCount = readImageFile(neg_imageFile, neg_image_path);

	//string image_folder = "C:/Users/lzm/Desktop/watermark_prepare/watermark_detection/neg_images";
	//getFiles(image_folder, image_path);
	/////////////////////////////////////////////////////////////////////////////////
	string featureFile = "toutiao_feature_hog_1056_50000_stage_2.xml";
	FileStorage fs_feature(featureFile, FileStorage::WRITE);
	int pos_use = 500000;
	int imageNum = min(pos_use, posCount) + negCount;
	int featureDim = 1056;
	fs_feature << "imageNum" << imageNum;
	fs_feature << "featureDim" << featureDim;
	fs_feature << "lable_feature" << "[";
	/*positive*/
	cout << "\n pos:  " << endl;
	for (int i = 0; i != min((size_t)pos_use, pos_image_path.size()); ++i)
	{
		if ((i + 1) % 100 == 0) cout << i+1 << endl;
		string temp = pos_image_path[i];
		Mat img = imread(temp,0);
		vector<float> feature;
		calculateHOG(img, feature);
		//calculateColorHOG(imgRGB, feature);
		fs_feature << 1 << feature;
	}
	/*negative*/
	cout << "\n neg: " << endl;
	for (int i = 0; i != neg_image_path.size(); ++i)
	{
		if ((i + 1) % 100 == 0) cout << i+1 << endl;
		Mat img = imread(neg_image_path[i],0);
		vector<float> feature;
		calculateHOG(img, feature);
		//calculateColorHOG(img, feature);
		fs_feature << -1 << feature;
	}
	fs_feature << "]";
	cout << featureFile << "   saved" << endl;
	return EXIT_SUCCESS;
}
