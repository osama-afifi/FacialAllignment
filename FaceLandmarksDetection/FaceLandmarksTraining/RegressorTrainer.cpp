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

void RegressorTrainer::regress(const vector<TrainingHelper::DataPoint> &training_data, vector<vector<Point2d> > &targets
							   ,const vector<Point2d> &mean_shape)
{
	for (int i = 0; i < config_setting.random_features_count; ++i)
	{
		// choose a random landmark
		selected_pixels[i].first = theRNG().uniform(0, training_data[0].landmarks.size());
		// generate some random noise where we will choose the pixel
		// May try gaussain vs uniform
		selected_pixels[i].second.x = theRNG().uniform(-config_setting.feature_random_dispersion, config_setting.feature_random_dispersion);
		selected_pixels[i].second.y = theRNG().uniform(-config_setting.feature_random_dispersion, config_setting.feature_random_dispersion);
	}

	// Remember pixel_val Matrix is (random features_count) x (training data size)
	Mat pixels_val(config_setting.random_features_count, training_data.size(), CV_64FC1);//, cv::alignPtr(pixels_val_data.get(), 32));

	for (int i = 0; i < training_data.size()  ; ++i) // i.e columns size
	{
		// Apply Procrustes Analysis (init->mean) to random offsets generated
		TrainingHelper::TransformMat trans_mat = TrainingHelper::procrustesAnalysis(training_data[i].init_shape, mean_shape);
		vector<cv::Point2d> offsets(config_setting.random_features_count);
		for (int j = 0; j < config_setting.random_features_count ; ++j)
			offsets[j] = selected_pixels[j].second;
		// Apply procrustes transform on passed offset
		trans_mat.apply(offsets, false); 
		
		for (int j = 0; j < config_setting.random_features_count ; ++j)
		{
			// calculate new pixel position after adding noise offset
			Point pixel_pos = training_data[i].init_shape[selected_pixels[j].first] 	+ offsets[j];
			if (pixel_pos.inside(cv::Rect(0, 0, training_data[i].image.cols, training_data[i].image.rows)))
				pixels_val.at<double>(j, i) = training_data[i].image.at<uchar>(pixel_pos);
			else
				//pixels_val.at<double>(j, i) = 0; // bad idea
				pixels_val.at<double>(j, i) = training_data[i].image.at<uchar>(Point(min(training_data[i].image.cols-1,max(pixel_pos.x,0)),
																					 min(training_data[i].image.rows-1,max(pixel_pos.y,0))
																					 ));
		}
	}

	Mat pixels_cov, means;
	cv::calcCovarMatrix(pixels_val, pixels_cov, means, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_COLS /* to use columns a input vectors*/);
	// pixels_cov is a (random_features_count x random_features_count) matrix 
	// means is column vector with the size of the train data
	for (int i = 0; i < config_setting.buttom_lvl_size; ++i)
	{
		//regress the new offset
		ferns[i].regress(targets, pixels_val, pixels_cov);
		// iterate over thetraining set and calculate the shape difference between 
		// the ground truth target shape and the regressd shape and hence update the new targets
		for (int j = 0; j < targets.size(); ++j)
			targets[j] = TrainingHelper::shapeDifference(targets[j], ferns[i].apply(pixels_val(cv::Range::all(), cv::Range(j, j + 1))));
	}
	// Reduce the Model Size by compressing ferns based on ...... TOCO
	//CompressFerns();

}

vector<Point2d> RegressorTrainer::apply(const vector<Point2d> &mean_shape, const TrainingHelper::DataPoint &data_point) const
{
	cv::Mat pixels_val(1, config_setting.random_features_count, CV_64FC1);
	TrainingHelper::TransformMat trans_mat = TrainingHelper::procrustesAnalysis(data_point.init_shape, mean_shape);
	vector<cv::Point2d> offsets(config_setting.random_features_count);
	for (int j = 0; j < config_setting.random_features_count ; ++j)
		offsets[j] = selected_pixels[j].second;
	trans_mat.apply(offsets, false);

	double *p = pixels_val.ptr<double>(0);
	for (int j = 0; j < config_setting.random_features_count ; ++j)
	{
		cv::Point pixel_pos = data_point.init_shape[selected_pixels[j].first] + offsets[j];
		if (pixel_pos.inside(cv::Rect(0, 0, data_point.image.cols, data_point.image.rows)))
			p[j] = data_point.image.at<uchar>(pixel_pos);
		else
			p[j] = 0;
	}

	//vector<double> coeffs(training_parameters.Base);
	//for (int i = 0; i < training_parameters.K; ++i)
	//	ferns[i].ApplyMini(pixels_val, coeffs);

	//cv::Mat result_mat = cv::Mat::zeros(mean_shape.size() * 2, 1, CV_64FC1);
	//for (int i = 0; i < training_parameters.Base; ++i)
	//	result_mat += coeffs[i] * base.col(i);
	//vector<cv::Point2d> result(mean_shape.size());
	//for (int i = 0; i < result.size(); ++i)
	//{
	//	result[i].x = result_mat.at<double>(i * 2);
	//	result[i].y = result_mat.at<double>(i * 2 + 1);
	//}
	//return result;
	return vector<Point2d>();
}
