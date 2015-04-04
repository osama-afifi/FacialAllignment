
#include "LandmarkTraining.h"
#include "RegressorTrainer.h"
#include <stdexcept>


using namespace std;

LandmarkTraining::LandmarkTraining(const string &training_dir, const string &output_model_dir)
	: training_dir(training_dir) , output_model_dir(output_model_dir)
{
}


LandmarkTraining::~LandmarkTraining(void)
{
}


void LandmarkTraining:: startTraining(void)
{
	readConfig(); // 1. Read Configuration File
	readData("/training",training_data); // 2. Read Training Data
	readData("/testing",testing_data); // 3. Read Testing Data
	createInitalShapes(); // 4. Create Initial Shapes
	createArgumentedData(); // 5. Argument Data
	vector<vector<cv::Point2d> > landmarks;
	for (const TrainingHelper::DataPoint &dp : training_data)
		landmarks.push_back(dp.landmarks);
	// 6. Calc Mean Shape
	mean_shape = TrainingHelper::meanShape(landmarks, config_setting);

	// 7. Start Cascaded Regression
	vector<RegressorTrainer> stage_regressors(config_setting.top_lvl_size, RegressorTrainer(config_setting));
	for (int i = 0; i < config_setting.top_lvl_size; ++i)
	{
		long long s = cv::getTickCount();

		vector<vector<cv::Point2d> > normalized_targets = computeNormalizedTargets();
		stage_regressors[i].regress(argumented_data, normalized_targets, mean_shape);
		// incrementally add offsets to the current regressed shape
		for (TrainingHelper::DataPoint &dp : argumented_data)
		{
			vector<cv::Point2d> offset = stage_regressors[i].apply(mean_shape, dp);
			TrainingHelper::TransformMat trans_mat = TrainingHelper::procrustesAnalysis(dp.init_shape, mean_shape);
			trans_mat.apply(offset, false);
			dp.init_shape = TrainingHelper::shapeAddition(dp.init_shape, offset);
		}
	}


}

vector<vector<cv::Point2d> > LandmarkTraining::computeNormalizedTargets()
{
	vector<vector<cv::Point2d> > norm_targets;
	for (const TrainingHelper::DataPoint& data_point : argumented_data)
	{
		vector<cv::Point2d> error = TrainingHelper::shapeDifference(data_point.landmarks, data_point.init_shape);
		TrainingHelper::TransformMat trans_mat = TrainingHelper::procrustesAnalysis(mean_shape, data_point.init_shape);
		trans_mat.apply(error, false);
		norm_targets.push_back(error);
	}
	return norm_targets;
}

void LandmarkTraining:: createArgumentedData()
{
	// You should provide training data with at least 2*ArgumentDataFactor images.
	assert(training_data.size() >= 2 * config_setting.argument_factor);
	argumented_data.clear();
	argumented_data.resize(training_data.size() * config_setting.argument_factor);
	for (int i = 0; i < training_data.size(); ++i)
	{
		set<int> landmark_indices;
		while (landmark_indices.size() <  2 * config_setting.argument_factor)
		{
			int rand_index = cv::theRNG().uniform(0, training_data.size());
			if (rand_index != i)
				landmark_indices.insert(rand_index);
		}
		auto iter = landmark_indices.cbegin();
		for (int j = i * config_setting.argument_factor; j < (i + 1) * config_setting.argument_factor; ++j, ++iter)
		{
			argumented_data[j] = training_data[i];
			argumented_data[j].init_shape = TrainingHelper::mapWindow(training_data[*iter].face_rect, training_data[*iter].landmarks, argumented_data[j].face_rect);
		}
	}
}

void LandmarkTraining:: createInitalShapes()
{
	if (config_setting.init_count > training_data.size())
		throw invalid_argument("TestInitShapeCount is larger than training image count, which is not allowed.");	
	const int landmarks_size = training_data[0].landmarks.size();
	cv::Mat flat_landmarks(training_data.size(), landmarks_size * 2, CV_32FC1);

	// flatten the landmarks coordinates for k-means
	for (int i = 0; i < training_data.size(); ++i)
	{
		// normalize the landmarks in [0,1] range
		vector<cv::Point2d> landmarks = TrainingHelper::mapWindow(training_data[i].face_rect, training_data[i].landmarks, cv::Rect(0, 0, 1, 1));
		for (int j = 0; j < landmarks_size; ++j)
		{
			flat_landmarks.at<float>(i, j * 2) = landmarks[j].x;
			flat_landmarks.at<float>(i, j * 2 + 1) = landmarks[j].y;
		}
	}

	// k-means on landmarks to clusters it into init_count clusters
	cv::Mat labels, centers;
	cv::kmeans(flat_landmarks, config_setting.init_count, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0), 10,
		cv::KMEANS_RANDOM_CENTERS | cv::KMEANS_PP_CENTERS, centers);

	// initialize the landmarks with centroids of k clusters instead 
	initial_shape_data.clear();
	for (int i = 0; i < config_setting.init_count; ++i)
	{
		vector<cv::Point2d> landmarks;
		for (int j = 0; j < landmarks_size ; ++j)
			landmarks.push_back(cv::Point2d(centers.at<float>(i, j * 2), centers.at<float>(i, j * 2 + 1)));
		initial_shape_data.push_back(landmarks);
	}
}

void LandmarkTraining:: readData(string sub_dir, std::vector<TrainingHelper::DataPoint> &result)
{
	result.clear();
	std::ifstream fin((training_dir + sub_dir + "/label.txt").c_str());
	if (!fin.fail())
	{
		std::string line;
		TrainingHelper:: DataPoint dataPoint;
		string current_path;
		cv::Point2d p;
		while (getline(fin, line))
		{
			fin>>current_path;
			dataPoint.image = cv::imread(training_dir + sub_dir + "/images"+current_path, CV_LOAD_IMAGE_GRAYSCALE);
			for(int i=0;i<LandmarkTraining::config_setting.landmark_count;++i)
			{
				fin>>p.x>>p.y;
				dataPoint.landmarks.push_back(p);
			}
			result.push_back(dataPoint);
		}
	}
	else
		throw std::runtime_error("Cannot open config file");
}

void LandmarkTraining:: readConfig(void)
{
	std::ifstream fin(training_dir.c_str());
	if (!fin.fail())
	{
		std::vector<std::string> items;
		std::string line;
		int line_no = 0;
		while (getline(fin, line))
		{
			++line_no;
			if (line.empty() || line[0] == '#')
				continue;
			int colon_pos = line.find(':');
			if (colon_pos == std::string::npos)
			{
				throw std::runtime_error("Illegal line " + std::to_string(line_no) +
				" in config file " );
			}
			items.push_back(line.substr(colon_pos));
		}
		config_setting.landmark_count = std::stoi(items[0]);
		config_setting.left_eye_index = std::stoi(items[1]);
		config_setting.right_eye_index = std::stoi(items[2]);
		config_setting.output_model_path = std::stoi(items[3]);
		config_setting.top_lvl_size = std::stoi(items[4]);
		config_setting.buttom_lvl_size = std::stoi(items[5]);
		config_setting.random_features_count = std::stoi(items[6]);
		config_setting.feature_random_dispersion = std::stoi(items[7]);
		config_setting.fern_depth = std::stoi(items[8]);
		config_setting.regular_coff = std::stoi(items[9]);
		config_setting.init_count = std::stoi(items[10]);
		config_setting.argument_factor = std::stoi(items[11]);
		//config_setting.high_lvl_count = std::stoi(items[12]);
		//config_setting.high_lvl_count = std::stoi(items[13]);
	}
	else
		throw std::runtime_error("Cannot open config file");
}

