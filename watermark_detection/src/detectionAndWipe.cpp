#include "detectionAndWipe.h"

void WatermarkSolve::processWatermark(const vector<string> &img_paths, vector<bool> &labels)
{
	if (labels.size() != img_paths.size())
		labels.resize(img_paths.size(), false);

	for (int i = 0; i != img_paths.size(); ++i)
	{
		if ((i + 1) % 1 == 0) cout << i + 1 << " images >" << endl;
		string tempPath = img_paths[i];
		Mat img = imread(tempPath);
		Mat imgGray = imread(tempPath, 0);
		if (!img.data || !imgGray.data)
		{
			cout << "can not read the image" << endl;
			continue;
		}
		if (img.cols < 100 || img.rows < 100) continue;
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		vector<Rect> strs;
		vector<int> rejectLevels;
		vector<double> levelWeights;
		double score = 0;
		watermark_pro->detect(imgGray, strs, rejectLevels, levelWeights, score, 1.1, 18);
		if (strs.empty()) continue;

		vector<double> s_1;
		watermark_pro->feature.hog_filter(imgGray, strs, 0, watermark_pro->boost, s_1);
		if (strs.empty()) continue;

		Rect res;
		double s = -100;
		for (int i = 0; i < strs.size(); ++i)
		{
			if (s_1[i] > s)
			{
				res = strs[i];
				s = s_1[i];
			}
		}

		Mat res_img(imgGray, res);
		s = watermark_pro->feature.hog_classify_one_boost(res_img, watermark_pro->boost_ii);
		if (s < 0) continue;

		labels[i] = true;

		watermark_pro->wipeWatermark(img, res);
		imwrite(tempPath, img);

	}
}

void WatermarkSolve::processWatermark_debug(const vector<string> &img_paths, vector<bool> &labels)
{
	if (labels.size() != img_paths.size())
		labels.resize(img_paths.size(), false);
	//ofstream score_o("score_out.txt");
	int count = 0;
	for (int i = 0; i != img_paths.size(); ++i)
	{
		if ((i + 1) % 1 == 0) cout << i + 1 << " images >" << endl;
		string tempPath = img_paths[i];
		Mat img = imread(tempPath); 
		Mat imgGray = imread(tempPath, 0);
		if (!img.data || !imgGray.data)
		{
			cout << "can not read the image" << endl;
			continue;
		}
		if (img.cols < 100 || img.rows < 100) continue;

		Mat show = img.clone();
		double t = getTickCount();
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		vector<Rect> strs;
		vector<int> rejectLevels;
		vector<double> levelWeights;
		double score = 0;
		Rect res=watermark_pro->detect(imgGray, strs, rejectLevels, levelWeights, score, 1.1, 18);
		if (strs.empty()) continue;
	
		vector<double> s_1;
		watermark_pro->feature.hog_filter(imgGray, strs, -50, watermark_pro->boost,s_1);
		if (strs.empty()) continue;

		vector<double> s_2;
		watermark_pro->feature.hog_filter(imgGray, strs, -50, watermark_pro->boost_ii, s_2);
		if (strs.empty()) continue;

		/*Mat res_img(imgGray, res);
		double s = watermark_pro->feature.hog_classify_one_boost(res_img, watermark_pro->boost);
		if (s < 0) continue;

		s = watermark_pro->feature.hog_classify_one_boost(res_img, watermark_pro->boost_ii);
		if (s < 0) continue;
		count++;*/
		
		///////////////////////////////////////////////////////////////////////////////////////////////////
		
	//	score_o << score << endl;
	//	rectsSaveAsImage(img, strs, tempPath, String("neg_hard_window"));

		labels[i] = true;

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "times passed for detection in ms: " << t*1000 << endl;

		t = getTickCount();
		watermark_pro->wipeWatermark(img, res);
		///*	for (int i=0;i!=strs.size();++i)
		//{
		//watermark_pro.wipeWatermark(img, strs[i]);
		//}*/

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "times passed for wipe in ms: " << t*1000 << endl;

		//watermark_pro->showResult(show, strs);
		rectangle(show, res, Scalar(0, 255, 255));
		int p = tempPath.find_first_of('.');
		int n = tempPath.find_first_of('\\');
		string fileName = tempPath.substr(n+1, p-n-1);
		string exten = tempPath.substr(p);

		string folder_name = "result_test/";

		t = getTickCount();

		imwrite((folder_name +fileName+ "-show" + exten), show);
		imwrite(folder_name + fileName+"-wipe" + exten, img);

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "times passed for save in ms: " << t * 1000 << endl;
	}
	//score_o.close();
	cout << "count:  " << count << endl;
}

vector<bool> WatermarkSolve::processWatermarkImpl(const vector<string> &img_paths)
{

	vector<bool> labels;

	
	//double t = getTickCount();

	processWatermark_debug(img_paths, labels);

	//t = ((double)getTickCount() - t) / getTickFrequency();
	//cout << "times passed for detection in ms: " << t * 1000 << endl;

	return labels;
}


