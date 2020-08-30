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
	//string pos_imageFile = "test_path.txt";
	vector<string> pos_image_path;
	int posCount = readImageFile(pos_imageFile, pos_image_path);

	string neg_imageFile = "../img_path/neg_stage_2_path.txt";
	vector<string> neg_image_path;
	int negCount = readImageFile(neg_imageFile, neg_image_path);

	//string image_folder = "C:/Users/lzm/Desktop/watermark_prepare/watermark_detection/neg_images";
	//getFiles(image_folder, image_path);
	/////////////////////////////////////////////////////////////////////////////////

	CvSVM svm;
	string svm_filename = "toutiao_feature_hog_1056_30000-SVM-l.xml";
	cout << "SVM file: " << svm_filename << endl;
	svm.load(svm_filename.c_str());

	CvBoost  boost;
	string boost_filename = "toutiao_feature_hog_1056_50000_stage_2-boost-300.xml";
	cout << "Boost file: " << boost_filename << endl;
	boost.load(boost_filename.c_str());

	/*positive*/
	cout << "\n pos:  " << endl;

	ofstream out_pos("pos_score.txt");
	int pos_count=0;
	for (int i = 0; i != min(size_t(100000),pos_image_path.size()); ++i)
	{
		if ((i + 1) % 100 == 0) cout << i+1 << endl;
		string temp = pos_image_path[i];
		Mat img = imread(temp,0);
		//double score = -hog_classify_one_svm(img, svm);
		//double score = -hog_color_classify_one_svm(imgRGB, svm);
		double score = hog_classify_one_boost(img, boost);
		if (score > 0) pos_count++;
		out_pos << score << endl;
	}
	out_pos.close();
	/*negative*/
	cout << "\n neg: " << endl;
	ofstream out_neg("neg_score.txt");
	int neg_fa_count=0;
	for (int i = 0; i != neg_image_path.size(); ++i)
	{
		if ((i + 1) % 100 == 0) cout << i+1 << endl;
		Mat img = imread(neg_image_path[i],0);
		Mat imgRGB = imread(neg_image_path[i]);
		//double score = -hog_classify_one_svm(img, svm);
		//double score = -hog_color_classify_one_svm(img, svm);
		double score = hog_classify_one_boost(img, boost);
		if (score > 0) neg_fa_count++;
		out_neg << score << endl;

		char recToSaved[500];
		sprintf(recToSaved, "%s/%d.png", "neg_hard", i);
		//if (score > 0) imwrite(recToSaved, imgRGB);
	}
	out_neg.close();
	cout << "pos_count: " << pos_count << "\nneg_fa: " << neg_fa_count << endl;
	return EXIT_SUCCESS;
}
