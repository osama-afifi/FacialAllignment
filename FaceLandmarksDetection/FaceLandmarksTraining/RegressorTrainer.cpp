#include "RegressorTrainer.h"

using namespace std;

RegressorTrainer::RegressorTrainer(TrainingHelper::ConfigParameters &config_setting) : config_setting(config_setting)
{
}


RegressorTrainer::~RegressorTrainer(void)
{
}



void RegressorTrainer::regress(const std::vector<cv::Point2d> &mean_shape, std::vector<std::vector<cv::Point2d> > &targets,
							   const std::vector<TrainingHelper::DataPoint> &training_data)
{

}
std::vector<cv::Point2d> RegressorTrainer::apply(const std::vector<cv::Point2d> &mean_shape, const TrainingHelper::DataPoint &data) const
{
	return vector<cv::Point2d>();
}
