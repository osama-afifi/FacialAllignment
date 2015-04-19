#include "FernTrainer.h"
#include "opencv/cv.h"
#include <vector>

using namespace std;
using namespace cv;

FernTrainer::FernTrainer(void)
{
}


FernTrainer::~FernTrainer(void)
{
}


void FernTrainer::regress(vector<vector<Point2d> > &targets, Mat pixels_val, Mat pixels_cov)
{
	// flatenning Y into 1 Channel
	cv::Mat Y(targets.size(), targets[0].size() * 2, CV_64FC1);
	for (int i = 0; i < Y.rows; ++i)
	{
		for (int j = 0; j < Y.cols; j += 2)
		{
			Y.at<double>(i, j) = targets[i][j / 2].x;
			Y.at<double>(i, j + 1) = targets[i][j / 2].y;
		}
	}
	
	features_index.assign(config_parameters.fern_depth, pair<int, int>());
	thresholds.assign(config_parameters.fern_depth, 0);

	for (int i = 0; i < config_parameters.fern_depth ; ++i)
	{
		// random gaussian projection values with mean 0 SD 1
		cv::Mat projection(Y.cols, 1, CV_64FC1);
		cv::theRNG().fill(projection, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(1));
		//unique_ptr<double[]> Y_proj_data(new double[Y.rows + 3]);
		//gogogogogogo
		cv::Mat Y_proj(Y.rows, 1, CV_64FC1);

		//project on targets
		static_cast<cv::Mat>(Y * projection).copyTo(Y_proj);

		double Y_proj_cov = TrainingHelper::Covariance(Y_proj,Y_proj);
		vector<double> Y_pixels_cov(pixels_val.rows);

		/////
		// (O-v-O)
		/////
		for (int j = 0; j < pixels_val.rows; ++j)
		{
			Y_pixels_cov[j] = TrainingHelper::Covariance(Y_proj, pixels_val);
		}

	//	double max_corr = -1;
	//	for (int j = 0; j < pixels_val.rows; ++j)
	//	{
	//		for (int k = 0; k < pixels_val.rows; ++k)
	//		{
	//			double corr = (Y_pixels_cov[j] - Y_pixels_cov[k]) / sqrt(Y_proj_cov * (pixels_cov.at<double>(j, j) + pixels_cov.at<double>(k, k) 
	//				- 2 * pixels_cov.at<double>(j, k)));
	//			if (corr > max_corr)
	//			{
	//				max_corr = corr;
	//				features_index[i].first = j;
	//				features_index[i].second = k;
	//			}
	//		}
	//	}

	//	double threshold_max = -1000000;
	//	double threshold_min = 1000000;
	//	for (int j = 0; j < pixels_val.cols; ++j)
	//	{
	//		double val = pixels_val.at<double>(features_index[i].first, j)
	//			- pixels_val.at<double>(features_index[i].second, j);
	//		threshold_max = max(threshold_max, val);
	//		threshold_min = min(threshold_min, val);
	//	}
	//	thresholds[i] = (threshold_max + threshold_min) / 2 
	//		+ cv::theRNG().uniform(-(threshold_max - threshold_min) * 0.1, 
	//		(threshold_max - threshold_min) * 0.1);
	}

	//int outputs_count = 1 << config_parameters.fern_depth;
	//outputs.assign(outputs_count, vector<cv::Point2d>((*targets)[0].size()));
	//vector<int> each_output_count(outputs_count);

	//for (int i = 0; i < targets.size(); ++i)
	//{
	//	int mask = 0;
	//	for (int j = 0; j < config_parameters.fern_depth; ++j)
	//	{
	//		double p1 = pixels_val.at<double>(features_index[j].first, i);
	//		double p2 = pixels_val.at<double>(features_index[j].second, i);
	//		mask |= (p1 - p2 > thresholds[j]) << j;
	//	}
	//	outputs[mask] = ShapeAdjustment(outputs[mask], (*targets)[i]);
	//	++each_output_count[mask];
	//}

	//for (int i = 0; i < outputs_count; ++i)
	//{
	//	for (cv::Point2d &p : outputs[i])
	//		p *= 1.0 / (each_output_count[i] + training_parameters.Beta);
	//}
}


vector<Point2d> FernTrainer::apply(Mat features)const
{
	return vector<Point2d>();
}
