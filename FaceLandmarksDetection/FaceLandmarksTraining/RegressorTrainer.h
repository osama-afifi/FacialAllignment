#include "TrainingHelper.h"
#include "FernTrainer.h"

#pragma once
class RegressorTrainer
{
public:
	RegressorTrainer(TrainingHelper::ConfigParameters &config_setting);
	~RegressorTrainer(void);

	void regress(const std::vector<TrainingHelper::DataPoint> &training_data, std::vector<std::vector<cv::Point2d> > &targets
							   , const std::vector<cv::Point2d> &mean_shape);
	std::vector<cv::Point2d> apply(const std::vector<cv::Point2d> &mean_shape, const TrainingHelper::DataPoint &data) const;

private:
	std::vector<std::pair<int, cv::Point2d> > pixels;
	std::vector<FernTrainer> ferns;
	cv::Mat base;
	const TrainingHelper::ConfigParameters &config_setting;

	void CompressFerns();
};

