
#include<string>
#include<vector>

#include "opencv/cv.h"

#pragma once
class TrainingHelper
{
public:
	
	TrainingHelper(void);
	~TrainingHelper(void);

	struct TransformMat;
	static std::vector<cv::Point2d> shapeDifference(const std::vector<cv::Point2d> &s1,const std::vector<cv::Point2d> &s2);
	static std::vector<cv::Point2d> mapWindow(cv::Rect original_rect, const std::vector<cv::Point2d> original_points, cv::Rect new_rect);
	static TrainingHelper::TransformMat procrustes(const std::vector<cv::Point2d> &x, const std::vector<cv::Point2d> &y);

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

	
	struct TransformMat
	{
		cv::Matx22d scale_rotation;
		cv::Matx21d translation;
		void apply(std::vector<cv::Point2d>&x, bool need_translation = true);
	};


};

