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


void FernTrainer::regress(vector<vector<Point2d> > &targets, Mat pixels_val, Mat pixels_covariance_matrix)
{
	// Flatening Y into 1 Channel
	cv::Mat targets_flat(targets.size(), targets[0].size() * 2, CV_64FC1);
	for (int i = 0; i < targets_flat.rows; ++i)
	{
		for (int j = 0; j < targets_flat.cols; j += 2)
		{
			targets_flat.at<double>(i, j) = targets[i][j / 2].x;
			targets_flat.at<double>(i, j + 1) = targets[i][j / 2].y;
		}
	}

	// initializing feature index and threshold vector of the fern
	features_index.assign(config_parameters.fern_depth, pair<int, int>());
	thresholds.assign(config_parameters.fern_depth, 0);

	for (int i = 0; i < config_parameters.fern_depth ; ++i)
	{
		// Fill projection with random gaussian values of mean 0 and standard deviation of 1
		cv::Mat projection(targets_flat.cols, 1, CV_64FC1);
		cv::theRNG().fill(projection, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(1));

		// Projected Landmarks initialization
		cv::Mat targets_projected(targets_flat.rows, 1, CV_64FC1);

		// Apply direction Projection on Landmarks Labels (Targets)
		static_cast<cv::Mat>(targets_flat * projection).copyTo(targets_projected);

		// Calculating the projected Targets covariance 
		double targets_projected_covariance = TrainingHelper::Covariance(targets_projected.ptr<double>(0),targets_projected.ptr<double>(0),targets_flat.rows);

		// Calculating the Covariance between projected targets landmark and each feature landmark
		vector<double> targets_pixels_covariance(pixels_val.rows);
		for (int j = 0; j < pixels_val.rows; ++j)
			targets_pixels_covariance[j] = TrainingHelper::Covariance(targets_projected.ptr<double>(0), pixels_val.ptr<double>(j,0), targets_projected.rows);

		// calculate features (pixels) with the maximum correlation 
			double max_correlation = -1;

		// Remember pixel_val Matrix is (random features_count) x (training data size)
			for (int j = 0; j < pixels_val.rows; ++j) 
			{
				for (int k = 0; k < pixels_val.rows; ++k)
				{
					// The Correlation between two features (pixels) [-1,+1]
				double correlation = (targets_pixels_covariance[j] - targets_pixels_covariance[k]) /
				 sqrt(targets_projected_covariance * (pixels_covariance_matrix.at<double>(j, j) + pixels_covariance_matrix.at<double>(k, k)
				 - 2 * pixels_covariance_matrix.at<double>(j, k)));
					if (correlation > max_correlation)
					{
						max_correlation = correlation;
						features_index[i].first = j;
						features_index[i].second = k;
					}
				}
			}

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
