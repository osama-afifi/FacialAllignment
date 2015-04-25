#pragma once
#include "TrainingHelper.h"
#include<memory>

class FernTrainer
{
public:
	FernTrainer(void);
	~FernTrainer(void);

	
	std::vector<double> threshold_vec;
	std::vector<std::pair<int, int> > features_pair_vec;
	std::vector<std::vector<cv::Point2d> > output_bucket;
	std::vector<std::vector<std::pair<int, double> > > outputs_mini;

	FernTrainer operator = (FernTrainer &rhs)
	{
		config_parameters = rhs.config_parameters;
		threshold_vec = rhs.threshold_vec;
		features_pair_vec = rhs.features_pair_vec;
		output_bucket= rhs.output_bucket;
		outputs_mini = rhs.outputs_mini;
		return *this;
	}

	void regress(std::vector<std::vector<cv::Point2d> > &targets, cv::Mat pixels_val, cv::Mat pixels_cov);
	std::vector<cv::Point2d> apply(cv::Mat features)const;
	
private:
	TrainingHelper::ConfigParameters config_parameters;
};

