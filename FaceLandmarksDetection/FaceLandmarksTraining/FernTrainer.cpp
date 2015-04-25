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
	features_pair_vec.assign(config_parameters.fern_depth, pair<int, int>());
	threshold_vec.assign(config_parameters.fern_depth, 0);

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
					// The Correlation between two features after direction projection (pixels) [-1,+1]
					// This is fast because our previous covariance computations
				double correlation = (targets_pixels_covariance[j] - targets_pixels_covariance[k]) /
				 sqrt(targets_projected_covariance * (pixels_covariance_matrix.at<double>(j, j) + pixels_covariance_matrix.at<double>(k, k)
				 - 2 * pixels_covariance_matrix.at<double>(j, k))
				 );
					if (correlation > max_correlation)
					{
						max_correlation = correlation;
						features_pair_vec[i].first = j;
						features_pair_vec[i].second = k;
					}
				}
			}

			double threshold_max = -1e9;
			double threshold_min = +1e9;
			for (int j = 0; j < pixels_val.cols; ++j)
			{
				double val = pixels_val.at<double>(features_pair_vec[i].first, j) - pixels_val.at<double>(features_pair_vec[i].second, j);
				threshold_max = max(threshold_max, val);
				threshold_min = min(threshold_min, val);
			}
			// The Middle of the Range as Threshold with some random noise
			threshold_vec[i] = (threshold_max + threshold_min) / 2 
			+ theRNG().uniform(-(threshold_max - threshold_min) * 0.1, (threshold_max - threshold_min) * 0.1);
	}

	int outputs_count = 1 << config_parameters.fern_depth;
	output_bucket.assign(outputs_count, vector<cv::Point2d>(targets[0].size()));
	vector<int> binary_string_freq(outputs_count,0);

	for (int i = 0; i < targets.size(); ++i)
	{
		int binary_mask = 0;
		// Building the binary string (fern bucket index) for each training data point
		for (int j = 0; j < config_parameters.fern_depth; ++j)
		{
			double p1 = pixels_val.at<double>(features_pair_vec[j].first, i);
			double p2 = pixels_val.at<double>(features_pair_vec[j].second, i);
			binary_mask |= ((p1 - p2) > threshold_vec[j]) << j;
		}
		// Incrementally add the shapes into the bucket
		output_bucket[binary_mask] = TrainingHelper::shapeAddition(output_bucket[binary_mask], targets[i]);
		++binary_string_freq[binary_mask];
	}

	// Averaging each bucket index according to it's number of shapes which map to it
	for (int i = 0; i < outputs_count; ++i)
	{
		for (cv::Point2d &point : output_bucket[i])
		{
			// Averaging
			point *= (1.0 / binary_string_freq[i]);
			// Regularization by multiplying by (1/(1+regular_coff/binary_freq[i])) 
			point *= (1.0 / (1.0 + (config_parameters.regular_coff/binary_string_freq[i])));
		}			
	}
}


vector<Point2d> FernTrainer::apply(Mat features)const
{
	int binary_mask = 0;
	for (int i = 0; i < config_parameters.fern_depth ; ++i)
	{
		double p1 = features.at<double>(features_pair_vec[i].first);
		double p2 = features.at<double>(features_pair_vec[i].second);
		binary_mask |= (p1 - p2 > threshold_vec[i]) << i;
	}
	return output_bucket[binary_mask];
}
