#include "LandmarkTraining.h"


LandmarkTraining::LandmarkTraining(const std::string &training_dir, const std::string &output_model_dir)
	: training_dir(training_dir) , output_model_dir(output_model_dir)
{
}


LandmarkTraining::~LandmarkTraining(void)
{
}


void LandmarkTraining:: startTraining(void)
{
	// 1. Read Configuration File

	// 2. Read Training Data

	// 3. Read Testing Data

	// 4. Create Initial Shapes

	// 5. Argument Data

	// 6. Start Cascaded Regression

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
		config_setting.high_lvl_count = std::stoi(items[4]);
		config_setting.low_lvl_count = std::stoi(items[5]);
		config_setting.random_features_count = std::stoi(items[6]);
		//config_setting.koppa = std::stoi(items[7]);
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


