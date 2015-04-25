#include<fstream>
#include <numeric>
#include <stdexcept>
#include <algorithm>

#include "TrainingHelper.h"

#include "opencv/cv.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"


#pragma once
class LandmarkTraining
{
private:

	std::string training_dir;
	std::string output_model_dir;
	TrainingHelper::ConfigParameters config_setting;
	std::vector<TrainingHelper::DataPoint> training_data;
	std::vector<TrainingHelper::DataPoint> argumented_data;
	std::vector<TrainingHelper::DataPoint> testing_data;
	std::vector<std::vector<cv::Point2d> > initial_shape_data;
	std::vector<cv::Point2d> mean_shape;

	std::vector<std::vector<cv::Point2d> > computeNormalizedTargets();
	void createInitalShapes();
	void createArgumentedData();
	void readConfig();
	void readData(std::string sub_dir, std::vector<TrainingHelper::DataPoint> &result);

public:

	LandmarkTraining(const std::string &training_dir, const std::string &output_model_dir);
	~LandmarkTraining(void);

	void startTraining();
	void saveModel();
	double getTrainingError();
	double getTestingError();
	
};

