#include "watermark.h"

WatermarkProcess::WatermarkProcess(Size _detect_size,
	double _h_ratio_min, 
	double _h_ratio_max, 
	bool _outputRejectLevels,
	double _roi_h_r, double _roi_w_r) :feature()
{
	roi_h_r = _roi_h_r;
	roi_w_r = _roi_w_r;

	detect_size = _detect_size;
	h_ratio_min = _h_ratio_min;
	h_ratio_max = _h_ratio_max;
	outputRejectLevels = _outputRejectLevels;
}

Rect WatermarkProcess::detect(const Mat& src,
	CV_OUT vector<Rect>& objects,
	vector<int>& rejectLevels,
	vector<double>& levelWeights,
	double &score,
	double scaleFactor,
	int minNeighbors)
{
	Mat img = src.clone();
	int object_h_min = img.rows*h_ratio_min;
	int object_h_max = img.rows*h_ratio_max;
	double s = (double)detect_size.height / object_h_min;
	if (object_h_min < detect_size.height)
	{
		resize(img, img, Size(), s, s, CV_INTER_LINEAR);
		object_h_min = img.rows*h_ratio_min;
		object_h_max = img.rows*h_ratio_max;
	}

	double h_r = (double)detect_size.width / detect_size.height;
	Size minSize = Size(object_h_min*h_r, object_h_min);
	Size maxSize = Size(object_h_max*h_r, object_h_max);

	Mat proImg = Mat(img, Rect(img.cols*roi_w_r, img.rows*roi_h_r, img.cols*(1-roi_w_r), img.rows*(1-roi_h_r))).clone();

	if (outputRejectLevels)
	{
		watermark_cascade.detectMultiScale(proImg, objects, rejectLevels, levelWeights, scaleFactor, minNeighbors, 0,
			minSize, maxSize, true);
	}
	else
	{
		watermark_cascade.detectMultiScale(proImg, objects, scaleFactor, minNeighbors, 0, minSize, maxSize);
	}

	for (int i=0;i<objects.size();++i)
	{
		Rect &r=objects[i];
		r = Rect(r.x + img.cols*roi_w_r, r.y + img.rows*roi_h_r, r.width, r.height);
		//rectangle(img, r, Scalar(0, 0, 255), 2);
	}

	if (s>1)
	{
		for (int i=0;i<objects.size();++i) 
		{
			Rect &r=objects[i];
			rectScale(r, 1 / s);
  		}
			
	}
	Rect res(0,0,0,0);
	double max_score=-10;
	if (!objects.empty() && outputRejectLevels)
	{
		res = objects[0];
		max_score = levelWeights[0];
		for (int i = 1; i != objects.size(); ++i)
		{
			if (levelWeights[i]>max_score)
			{
				max_score = levelWeights[i];
				res = objects[i];
			}
		}
	}
	score = max_score;

	res.x = max(0, res.x);
	res.y = max(0, res.y);
	res.width = res.x + res.width > src.cols ? src.cols - res.x : res.width;
	res.height = res.y + res.height > src.rows ? src.rows - res.y : res.height;
	return res;
}

void WatermarkProcess::showResult(Mat &img, const vector<Rect> &strs)
{
	for (int i=0;i<strs.size();++i)
	{
		Rect r=strs[i];
		rectangle(img, r, Scalar(0, 0, 255), 2);
	}
}

void WatermarkProcess::rectScale(Rect &r, double s)
{
	r.x = cvRound(r.x*s);
	r.y = cvRound(r.y*s);
	r.width = cvRound(s*r.width);
	r.height = cvRound(s*r.height);
}

void WatermarkProcess::wipeWatermark(Mat &img, Rect &res)
{
	Mat mask(img.size(), CV_8UC1, Scalar(0));
	Rect r(res);
	r.x -= 3;
	r.y -= 3;
	r.width = img.cols - r.x-5;
	r.height = min(r.height + 5, img.rows - r.y);

	rectangle(mask, r, Scalar(255), -1);
	inpaint(img, mask, img, 3, INPAINT_TELEA);
}
