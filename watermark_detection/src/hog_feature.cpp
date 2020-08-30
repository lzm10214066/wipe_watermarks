#include "hog_feature.h"

HOG_feature::HOG_feature(int _nbins,
	Size _blockSize,
	Size _blockStride,
	Size _NormSize,
	bool _color,
	double _s_th)
	{
		nbins = _nbins;
		blockSize = _blockSize;
		blockStride = _blockStride;
		NormSize = _NormSize;
		color = _color;
		s_th = _s_th;
}

void HOG_feature::calculateHOG(Mat &src, vector<float> &descriptors)
{
	Size cellSize(blockSize.width / 2, blockSize.height / 2);
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

double HOG_feature::hog_classify_one_svm(Mat &img, CvSVM &svm)
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

void HOG_feature::hog_filter(Mat &img, vector<Rect> &strs, double th, CvSVM &svm,bool color)
{
	if (!color)
	{
		vector<Rect> res;
		for (int i=0;i<strs.size();++i)
		{
			Rect r=strs[i];
			if (r.x<0 || r.x + r.width>img.cols || r.y<0 || r.y + r.height>img.rows) continue;
			Mat roi = Mat(img, r).clone();
			double s = -hog_classify_one_svm(roi, svm);
			if (s>th)
				res.push_back(r);
		}
		strs.clear();
		strs = res;
	}
	else
	{
		vector<Rect> res;
		for (int i=0;i<strs.size();++i)
		{
			Rect r=strs[i];
			if (r.x<0 || r.x + r.width>img.cols || r.y<0 || r.y + r.height>img.rows) continue;
			Mat roi = Mat(img, r).clone();
			double s = -hog_color_classify_one_svm(roi, svm);
			if (s>th)
				res.push_back(r);
		}
		strs.clear();
		strs = res;
	}
}

void HOG_feature::hog_filter(Mat &img, vector<Rect> &strs, double th, CvSVM &svm, bool color,vector<double> &scores)
{
	if (!color)
	{
		vector<Rect> res;
		for (int i=0;i<strs.size();++i)
		{
			Rect r=strs[i];
			if (r.x<0 || r.x + r.width>img.cols || r.y<0 || r.y + r.height>img.rows) continue;
			Mat roi = Mat(img, r).clone();
			double s = -hog_classify_one_svm(roi, svm);
			if (s > th)
			{
				scores.push_back(s);
				res.push_back(r);
			}
				
		}
		strs.clear();
		strs = res;
	}
	else
	{
		vector<Rect> res;
		for (int i=0;i<strs.size();++i)
		{
			Rect r=strs[i];
			if (r.x<0 || r.x + r.width>img.cols || r.y<0 || r.y + r.height>img.rows) continue;
			Mat roi = Mat(img, r).clone();
			double s = -hog_color_classify_one_svm(roi, svm);
			if (s > th)
			{
				scores.push_back(s);
				res.push_back(r);
			}
		}
		strs.clear();
		strs = res;
	}

}

void HOG_feature::calculateColorHOG(Mat &src, vector<float> &descriptors)
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

double HOG_feature::hog_color_classify_one_svm(Mat &img, CvSVM &svm)
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

double HOG_feature::hog_classify_one_boost(Mat &img, CvBoost &boost)
{
	vector<float> hist;
	calculateHOG(img, hist);

	Mat hist_mat(1, hist.size(), CV_32FC1);
	for (int i = 0; i < hist.size(); ++i)
	{
		hist_mat.at<float>(0, i) = hist[i];
	}
	return(boost.predict(hist_mat, Mat(), Range::all(), false, true));
}


void HOG_feature::hog_filter(Mat &img, vector<Rect> &strs, double th, CvBoost &boost, vector<double> &scores)
{
	
		vector<Rect> res;
		for (int i = 0; i<strs.size(); ++i)
		{
			Rect r = strs[i];
			if (r.x<0 || r.x + r.width>img.cols || r.y<0 || r.y + r.height>img.rows) continue;
			Mat roi = Mat(img, r).clone();
			double s = hog_classify_one_boost(roi, boost);
			if (s > th)
			{
				scores.push_back(s);
				res.push_back(r);
			}

		}
		strs.clear();
		strs = res;
}
