#include "RegressorTrainer.h"
#include "TrainingHelper.h"

using namespace std;
using namespace cv;

RegressorTrainer::RegressorTrainer(TrainingHelper::ConfigParameters &config_setting) : config_setting(config_setting)
{
}


RegressorTrainer::~RegressorTrainer(void)
{
}



void RegressorTrainer::regress(const std::vector<TrainingHelper::DataPoint> &training_data, std::vector<std::vector<cv::Point2d> > &targets
							   ,const std::vector<cv::Point2d> &mean_shape)
{
	for (int i = 0; i < config_setting.random_features_count; ++i)
	{
		// choose a random landmark
		pixels[i].first = cv::theRNG().uniform(0, training_data[0].landmarks.size());
		// generate some random noise where we will choose the pixel
		// May try gaussain vs uniform
		pixels[i].second.x = cv::theRNG().uniform(-config_setting.feature_random_dispersion, config_setting.feature_random_dispersion);
		pixels[i].second.y = cv::theRNG().uniform(-config_setting.feature_random_dispersion, config_setting.feature_random_dispersion);
	}

	// unflatten features here (not sure why)
	//unique_ptr<double[]> pixels_val_data(new double[training_parameters.P * training_data.size() + 3]);

	cv::Mat pixels_val(config_setting.random_features_count, training_data.size(), CV_64FC1);//, cv::alignPtr(pixels_val_data.get(), 32));

	for (int i = 0; i < training_data.size()  ; ++i) // i.e columns size
	{
		// Apply Procrustes Analysis to random offsets generated
		TrainingHelper::TransformMat trans_mat = TrainingHelper::procrustesAnalysis(training_data[i].init_shape, mean_shape);
		vector<cv::Point2d> offsets(config_setting.random_features_count);

		for (int j = 0; j < config_setting.random_features_count ; ++j)
			offsets[j] = pixels[j].second;
		trans_mat.apply(offsets, false); // apply procrustes transform on passed offset
		
		for (int j = 0; j < config_setting.random_features_count ; ++j)
		{
			cv::Point pixel_pos = training_data[i].init_shape[pixels[j].first] 	+ offsets[j];
			if (pixel_pos.inside(cv::Rect(0, 0, training_data[i].image.cols, training_data[i].image.rows)))
			{
				pixels_val.at<double>(j, i) = training_data[i].image.at<uchar>(pixel_pos);
			}
			// bad idea
			else
				pixels_val.at<double>(j, i) = 0;
		}
	}

	cv::Mat pixels_cov, means;
	cv::calcCovarMatrix(pixels_val, pixels_cov, means, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_COLS);

	//for (int i = 0; i < config_setting.buttom_lvl_size; ++i)
	//{
	//	ferns[i].regress(targets, pixels_val, pixels_cov);
	//	for (int j = 0; j < targets.size(); ++j)
	//	{
	//		(*targets)[j] = TrainingHelper::shapeDifference((*targets)[j], ferns[i].apply(pixels_val(cv::Range::all(), cv::Range(j, j + 1))));
	//	}
	//}
	//CompressFerns();

}
std::vector<cv::Point2d> RegressorTrainer::apply(const std::vector<cv::Point2d> &mean_shape, const TrainingHelper::DataPoint &data) const
{
	return vector<cv::Point2d>();
}
