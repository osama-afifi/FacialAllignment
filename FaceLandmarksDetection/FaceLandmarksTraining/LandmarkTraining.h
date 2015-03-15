#include<fstream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include "TrainingHelper.h"

#pragma once
class LandmarkTraining
{
private:
	std::string training_dir;
	std::string output_model_dir;
	TrainingHelper::ConfigParameters config_setting;
	std::vector<TrainingHelper::DataPoint> training_data;
	std::vector<TrainingHelper::DataPoint> testing_data;


	void readConfig();
	void readTrainingData();
	void readTestingData();


public:
	LandmarkTraining(const std::string &training_dir, const std::string &output_model_dir);
	~LandmarkTraining(void);

	void startTraining();
	void saveModel();
	double getTrainingError();
	double getTestingError();
	
};

