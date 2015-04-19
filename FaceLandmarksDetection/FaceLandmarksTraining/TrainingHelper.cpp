#include <assert.h>
#include "TrainingHelper.h"

using namespace std;
using namespace cv;

TrainingHelper::TrainingHelper(void)
{
}


TrainingHelper::~TrainingHelper(void)
{
}

vector<cv::Point2d> TrainingHelper::shapeDifference(const vector<cv::Point2d> &s1,const vector<cv::Point2d> &s2)
{
	assert(s1.size() == s2.size());
	vector<cv::Point2d> diff(s1.size());
	for (int i = 0; i < s1.size(); ++i)
		diff[i] = s1[i] - s2[i];
	return diff;
}


vector<cv::Point2d> TrainingHelper::mapWindow(cv::Rect original_rect, const vector<cv::Point2d> original_points, cv::Rect new_rect)
{
	vector<cv::Point2d> mappedPoints;
	for (const cv::Point2d &landmark: original_points)
	{
		mappedPoints.push_back(landmark);
		mappedPoints.back() -= cv::Point2d(original_rect.x, original_rect.y);
		mappedPoints.back().x *= (double)(new_rect.width) / original_rect.width;
		mappedPoints.back().y *= (double)(new_rect.height) / original_rect.height;
		mappedPoints.back() += cv::Point2d(new_rect.x, new_rect.y);
	}
	return mappedPoints;
}

// Returns a Transform Matrix that turns one shape closest to another
TrainingHelper::TransformMat TrainingHelper::procrustesAnalysis(const vector<cv::Point2d> &x, const vector<cv::Point2d> &y)
{
	assert(x.size() == y.size());
	int landmark_count = x.size();
	double X1 = 0, X2 = 0, Y1 = 0, Y2 = 0, Z = 0, W = landmark_count;
	double C1 = 0, C2 = 0;
	for (int i = 0; i < landmark_count; ++i)
	{
		X1 += x[i].x;
		X2 += y[i].x;
		Y1 += x[i].y;
		Y2 += y[i].y;
		Z += (y[i].x)*(y[i].x) + (y[i].y)*(y[i].y);
		C1 += x[i].x * y[i].x + x[i].y * y[i].y;
		C2 += x[i].y * y[i].x - x[i].x * y[i].y;
	}

	cv::Matx44d A(X2, -Y2,  W,  0,
				  Y2,  X2,  0,  W,
				  Z,   0,  X2,  Y2,
				  0,   Z, -Y2,  X2);
	cv::Matx41d b(X1, Y1, C1, C2);
	cv::Matx41d solution = A.inv() * b;

	TrainingHelper::TransformMat trans_mat;
	trans_mat.scale_rotation(0, 0) = solution(0);
	trans_mat.scale_rotation(0, 1) = -solution(1);
	trans_mat.scale_rotation(1, 0) = solution(1);
	trans_mat.scale_rotation(1, 1) = solution(0);
	trans_mat.translation(0) = solution(2);
	trans_mat.translation(1) = solution(3);
	return trans_mat;
}

void TrainingHelper::TransformMat::apply(vector<cv::Point2d> &shape, bool need_translation)
{
	for(int i = 0 ; i < shape.size() ; i++)
	{
		cv::Matx21d colVec;
		colVec(0) = shape[i].x;
		colVec(1) = shape[i].y;
		colVec = scale_rotation * colVec;
		if (need_translation)
			colVec += translation;
		shape[i].x = colVec(0);
		shape[i].y = colVec(1);
	}
}

std::vector<cv::Point2d> TrainingHelper::shapeAddition(const std::vector<cv::Point2d> &shape, const std::vector<cv::Point2d> &offset)
{
	assert(shape.size() == offset.size());
	vector<cv::Point2d> sum_shape(shape.size());
	for (int i = 0; i < shape.size(); ++i)
		sum_shape[i] = shape[i] + offset[i];
	return sum_shape;
}


std::vector<cv::Point2d> TrainingHelper::meanShape(std::vector<std::vector<cv::Point2d> > shapes, ConfigParameters &config_setting)
{
	// Iterations for averaging the shapes
	const int kIterationCount = 10;
	vector<cv::Point2d> mean_shape = shapes[0];
	
	for (int i = 0; i < kIterationCount; ++i)
	{
		for (vector<cv::Point2d> &shape: shapes)
		{
			TrainingHelper::TransformMat trans_mat = procrustesAnalysis(mean_shape, shape);
			trans_mat.apply(shape);
		}		
		for (int j = 0; j < mean_shape.size(); ++j)
			mean_shape[j].x = mean_shape[j].y = 0;		
		for (const vector<cv::Point2d> & shape : shapes)
			for (int j = 0; j < mean_shape.size(); ++j)
			{
				mean_shape[j].x += shape[j].x;
				mean_shape[j].y += shape[j].y;
			}
		for (cv::Point2d & p : mean_shape)
			p *= 1.0 / shapes.size();
		normalizeShape(mean_shape, config_setting);
	}
	return mean_shape;
}


// subtracting the centroid on the eye distance scale rotated to be an alligned face
void TrainingHelper::normalizeShape(vector<cv::Point2d> &shape, const TrainingHelper::ConfigParameters &config_setting)
{

	cv::Point2d center;
	for (const cv::Point2d &p : shape)
		center += p;
	center *= 1.0 / shape.size();
	for (cv::Point2d &p : shape)
		p -= center;

	cv::Point2d left_eye = shape.at(config_setting.left_eye_index);
	cv::Point2d right_eye = shape.at(config_setting.right_eye_index);
	double eyes_distance = cv::norm(left_eye - right_eye);
	double scale = 1.0 / eyes_distance;

	double theta = -atan((right_eye.y - left_eye.y) / (right_eye.x - left_eye.x));

	// Must do translation first, and then rotation.
	// Therefore, translation is done separately
	TransformMat trans_mat;
	trans_mat.scale_rotation(0, 0) = scale * cos(theta);
	trans_mat.scale_rotation(0, 1) = -scale * sin(theta);
	trans_mat.scale_rotation(1, 0) = scale * sin(theta);
	trans_mat.scale_rotation(1, 1) = scale * cos(theta);
	trans_mat.apply(shape, false);
}


	double TrainingHelper::Covariance(const vector<double> &x,const vector<double> &y)
{
	assert(x.size() && x.size()==y.size());
	double a = 0, b = 0, c = 0, dsize = x.size();
	for (int i = 0; i < x.size(); ++i)
	{
		a += x[i];
		b += y[i];
		c += x[i] * y[i];
	}

	return (c / dsize - (a / dsize) * (b / dsize));
}

	double TrainingHelper::Covariance(const  Mat x,const Mat y)
{
	assert(x.rows && x.rows == y.rows);
	double a = 0, b = 0, c = 0, dsize = x.rows;
	for (int i = 0; i < x.rows ; ++i)
	{
		a += x.at<double>(i);
		b += y.at<double>(i);
		c += x.at<double>(i) * y.at<double>(i);
	}

	return (c / dsize - (a / dsize) * (b / dsize));
}