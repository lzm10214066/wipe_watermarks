#ifndef _DETECTIONANDWIPE_H
#define _DETECTIONANDWIPE_H

#include "hog_feature.h"
#include "watermark.h"
#include "imageRead.h"
#include "opencv2/ml/ml.hpp"
#include "opencv2/objdetect/objdetect.hpp"

struct WatermarkSolve
{
	Ptr<WatermarkProcess> watermark_pro;
	WatermarkSolve()
	{
		const string cascade_classifier = "classifier/cascade_hog_3scales_new_20.xml";
		const string hog_classifier = "classifier/toutiao_feature_hog_1056_50000-boost.xml";
		const string hog_classifier_II = "classifier/toutiao_feature_hog_1056_50000_stage_2-boost-300.xml";

		Size detect_size = Size(42, 16);
		double h_ratio_max = 0.15;   //0.06
		double h_ratio_min = 0.035; //0.035
		bool outputRejectLevels = true;
		double roi_h_r = 0.8;  //0.8
		double roi_w_r = 0.2;  //0.1

		watermark_pro = new WatermarkProcess(detect_size, h_ratio_min, h_ratio_max, outputRejectLevels, roi_h_r, roi_w_r);
		if (!watermark_pro->watermark_cascade.load(cascade_classifier)){ cout << ("--(!)Error loading\n"); };
		//watermark_pro->svm.load(hog_classifier.c_str());
		watermark_pro->boost.load(hog_classifier.c_str());
		watermark_pro->boost_ii.load(hog_classifier_II.c_str());

	}

	vector<bool> processWatermarkImpl(const vector<string> &img_paths);
	void processWatermark(const vector<string> &img_paths, vector<bool> &labels);
	void processWatermark_debug(const vector<string> &img_paths, vector<bool> &labels);
};


#endif