#include "convolution.cpp"


int main(int argc, char *argv[])
{


	cv::Mat input = cv::imread(argv[1], 0);

	std::ifstream fs1(argv[2]);

	std::string str;
	std::string num;
	int filter_rows = 0;
	int filter_cols;


	while(getline(fs1, str))
	{
  
		std::stringstream ss(str);
		filter_rows++;

		filter_cols = 0;

		while(getline(ss, num, ','))
		{

			filter_cols++;
		}

	}



	std::vector< std::vector<double> > filter(filter_rows, std::vector<double> (filter_cols));

	int y = 0;

	std::ifstream fs2(argv[2]);


	while(getline(fs2, str))
	{

		std::stringstream ss(str);
		int x = 0;

		while(getline(ss, num, ','))
		{

			filter[y][x] = std::stod(num);
			x++;
		}

		y++;

	}




	clock_t start = clock();
	cv::Mat output1 = convolution_normal(input, filter);
	std::cout << "convolution_normal : " << (double)clock()-start << std::endl;

	start = clock();
	cv::Mat output2 = convolution_fft(input, filter);
	std::cout << "convolution_fft : " << (double)clock()-start << std::endl;


	double minVal, maxVal;
	cv::minMaxLoc(output1, &minVal, &maxVal);
	output1 -= minVal;
	int scale = (maxVal - minVal + 1) / 255;
	output1 /= scale;

	cv::minMaxLoc(output2, &minVal, &maxVal);
	output2 -= minVal;
	scale = (maxVal - minVal + 1) / 255;
	output2 /= scale;


	std::string output_normal("normal_");
	output_normal += argv[3];
	std::string output_fft("fft_");
	output_fft += argv[3];

	cv::imwrite(output_normal.c_str(), output1);
	cv::imwrite(output_fft.c_str(), output2);

	return 0;

}