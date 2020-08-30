#include <stdio.h>
#include <stdlib.h>
#include <io.h>

#include <iostream>
#include <fstream>

#include <opencv2\opencv.hpp>

#include "../src/imageRead.h"
#include "../src/utility.h"
#include "../src/watermark.h"
#include "../src/hog_feature.h"

#include "../src/detectionAndWipe.h"

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
	//////////////////////////////////////*read path.txt*///////////////////////////
	vector<string> image_path;

	string imageFile = "neg_path.txt";
	//int imageCount = readImageFile(imageFile, image_path);
	//if (argc < 2)
	//{
	//	cout << "There is no folder" << endl;
	//	//return -1;
	//}
	//string image_folder(argv[1]);
	//string image_folder = "watermark_test";
	
//	string image_folder = "I:/TEMP/20161123_0.0vs1.0";
	string image_folder = "test";
	getFiles(image_folder, image_path);

	/////////////////////////////////////////////////////////////////////////////////
	WatermarkSolve ws;
	vector<bool> labels = ws.processWatermarkImpl(image_path);


	return EXIT_SUCCESS;
}
