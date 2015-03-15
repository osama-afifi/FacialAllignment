#include <assert.h>
#include "TrainingHelper.h"

using namespace std;

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

TrainingHelper::TransformMat TrainingHelper::procrustes(const vector<cv::Point2d> &x, const vector<cv::Point2d> &y)
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

void TrainingHelper::TransformMat::apply(vector<cv::Point2d> &x, bool need_translation)
{
	for (cv::Point2d &p : x)
	{
		cv::Matx21d v;
		v(0) = p.x;
		v(1) = p.y;
		v = scale_rotation * v;
		if (need_translation)
			v += translation;
		p.x = v(0);
		p.y = v(1);
	}
}
