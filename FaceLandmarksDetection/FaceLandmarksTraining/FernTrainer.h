#pragma once
#include "TrainingHelper.h"
#include<memory>

class FernTrainer
{
public:
	FernTrainer(void);
	~FernTrainer(void);

	
	std::vector<double> thresholds;
	std::vector<std::pair<int, int> > features_index;
	std::vector<std::vector<cv::Point2d> > outputs;
	std::vector<std::vector<std::pair<int, double> > > outputs_mini;
	FernTrainer operator = (FernTrainer &rhs)
	{
		config_parameters = rhs.config_parameters;
		thresholds = rhs.thresholds;
		features_index = rhs.features_index;
		outputs= rhs.outputs;
		outputs_mini = rhs.outputs_mini;
		return *this;
	}

	void regress(std::vector<std::vector<cv::Point2d> > &targets, cv::Mat pixels_val, cv::Mat pixels_cov);
	std::vector<cv::Point2d> apply(cv::Mat features)const;
	//void apply();
private:
	TrainingHelper::ConfigParameters config_parameters;
};

