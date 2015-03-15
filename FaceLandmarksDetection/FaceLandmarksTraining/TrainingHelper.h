#include<string>
#include<vector>
#include "opencv/cv.h"

#pragma once
static class TrainingHelper
{
public:
	TrainingHelper(void);
	~TrainingHelper(void);

struct ConfigParameters
{
	std::string output_model_path;
	int landmark_count;
	int left_eye_index;
	int right_eye_index;
	int high_lvl_count;
	int low_lvl_count;
	int random_features_count;
	//double Kappa;
	int fern_depth;
	int regular_coff;
	int init_count;
	int argument_factor;
	//int Base; maybe for PCA
	//int Q;
};

struct DataPoint
{
	cv::Mat image;
	cv::Rect face_rect;
	std::vector<cv::Point2d> landmarks;
	std::vector<cv::Point2d> init_shape;
};




};

