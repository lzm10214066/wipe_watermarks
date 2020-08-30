#include "feature.h"

void calculateHOG(Mat &src, vector<float> &descriptors)
{
	int nbins = 8;
	Size blockSize(8, 8);
	Size blockStride(4, 4);
	Size cellSize(blockSize.width/2, blockSize.height/2);
	Size NormSize(48, 16);

	bool gammaCorrection = false;
	double winSigma = 4;
	double L2HysThreshold = 0.2;

	int featureDim = nbins * (blockSize.width / cellSize.width*blockSize.height / cellSize.height) *
		((NormSize.width - blockSize.width) / blockStride.width + 1) * ((NormSize.height - blockSize.height) / blockStride.height + 1);

	HOGDescriptor *hog = new cv::HOGDescriptor(NormSize, blockSize, blockStride, cellSize, nbins, 1, winSigma,
		HOGDescriptor::L2Hys, L2HysThreshold, gammaCorrection);

	if (src.channels() != 1) cvtColor(src, src, CV_BGR2GRAY);

	Mat imgNorm(NormSize, CV_8UC1);
	resize(src, imgNorm, NormSize);

	hog->compute(imgNorm, descriptors, Size(8, 8), Size(0, 0));
}

void calculateColorHOG(Mat &src, vector<float> &descriptors)
{
	CV_Assert(src.channels() == 3);

	Mat src_hsv;
	cvtColor(src, src_hsv, CV_BGR2HSV);
	vector<Mat> hsv;
	split(src_hsv, hsv);
	Mat h = hsv.at(0);
	Mat s = hsv.at(1);
	Mat v = hsv.at(2);

	vector<float> h_d, s_d, v_d;
	calculateHOG(h, h_d);
	calculateHOG(s, s_d);
	calculateHOG(v, v_d);

	for (int i = 0; i < 33; ++i)
	{
		vector<float> temp;
		for (int j = 0; j < 32; ++j)
		{
			temp.push_back(h_d[i * 32 + j]);
			temp.push_back(s_d[i * 32 + j]);
			temp.push_back(v_d[i * 32 + j]);
		}
		//normalize(temp, temp, 1, 0, NORM_L2);
		descriptors.insert(descriptors.end(), temp.begin(), temp.end());
	}
}

double hog_classify_one_svm(Mat &img, CvSVM &svm)
{
	vector<float> hist;
	calculateHOG(img, hist);

	Mat hist_mat(1, hist.size(), CV_32FC1);
	for (int i = 0; i < hist.size(); ++i)
	{
		hist_mat.at<float>(0, i) = hist[i];
	}
	return(svm.predict(hist_mat, true));
}

double hog_color_classify_one_svm(Mat &img, CvSVM &svm)
{
	vector<float> hist;
	calculateColorHOG(img, hist);

	Mat hist_mat(1, hist.size(), CV_32FC1);
	for (int i = 0; i < hist.size(); ++i)
	{
		hist_mat.at<float>(0, i) = hist[i];
	}
	return(svm.predict(hist_mat, true));
}

double hog_classify_one_boost(Mat &img,CvBoost &boost)
{
	vector<float> hist;
	calculateHOG(img, hist);

	Mat hist_mat(1, hist.size(), CV_32FC1);
	for (int i = 0; i < hist.size(); ++i)
	{
		hist_mat.at<float>(0, i) = hist[i];
	}
	return(boost.predict(hist_mat, Mat(), Range::all(),false, true));
}

