#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <math.h>
#include <fstream>
#include <time.h>



cv::Mat convolution_normal(cv::Mat input, std::vector< std::vector<double> > filter)
{


	int filter_rows = filter.size();
	int filter_cols = filter[0].size();


	cv::Mat output = cv::Mat::zeros(input.rows, input.cols, CV_32FC1);


	for(int y1=0;y1<input.rows;y1++)
	{
		for(int x1=0;x1<input.cols;x1++)
		{
			for(int y2=0;y2<filter_rows;y2++)
			{
				for(int x2=0;x2<filter_cols;x2++)
				{
			
					int pos_x = x1-(filter_cols/2)+x2;
					int pos_y = y1-(filter_rows/2)+y2;

			
					if(pos_x >= 0 && pos_x < input.cols && pos_y >= 0 && pos_y < input.rows)
					{

						float px = input.data[pos_y * input.cols + pos_x]
							* filter[filter_rows-y2-1][filter_cols-x2-1];

						output.at<float>(y1, x1) += px;

					}

				}
			}
		}
	}


	return output;

}





std::complex<double> W(double num, bool p){

	if(p)
	{
		return std::exp(std::complex<double> (0,1) * std::complex<double> (-2*M_PI*num,0));
	}
	else
	{
		return std::exp(std::complex<double> (0,1) * std::complex<double> (2*M_PI*num,0));
	}

}




std::vector< std::vector< std::complex<double> > > fft(std::vector< std::vector< std::complex<double> > > f, bool p)
{


	int width = f.size();
	int d = log2(width);


	std::vector< std::vector< std::complex<double> > > F(width, std::vector< std::complex<double> >(width));



	for(int m=0;m<d-1;m++)
	{
		for(int y=0;y<width;y++)
		{
			for(int x=0;x<width;x++)
			{
			
				int v = floor(y/pow(2,d-m))*pow(2,d-m) + y%(int)pow(2,d-m-1)*2 + (int)floor(y/pow(2,d-m-1))%2;
				int u = floor(x/pow(2,d-m))*pow(2,d-m) + x%(int)pow(2,d-m-1)*2 + (int)floor(x/pow(2,d-m-1))%2;			

				F[y][x] = f[v][u];
			}
		}
	
		f = F;
	}



	

	for(int m=0;m<d;m++)
	{
		for(int v=0;v<width;v++)
		{
			for(int u=0;u<width;u++)
			{

				int y = floor(v/pow(2,m+1))*pow(2,m+1) + v%(int)pow(2,m);
				int x = floor(u/pow(2,m+1))*pow(2,m+1) + u%(int)pow(2,m);
				int l = pow(2,m);
				double j = v%(int)pow(2,m+1);
				double k = u%(int)pow(2,m+1);

				F[v][u] = f[y][x]
						+ f[y][x+l] * W(k/pow(2,m+1), p)
						+ f[y+l][x] * W(j/pow(2,m+1), p)
						+ f[y+l][x+l] * W((j+k)/pow(2,m+1), p);
			}
		}

		f = F;

	}


	return f;

}




cv::Mat convolution_fft(cv::Mat input, std::vector< std::vector<double> > filter)
{


	int filter_rows = filter.size();
	int filter_cols = filter[0].size();


	int width = std::max({input.rows, input.cols, filter_rows, filter_cols});
	int d = 0;

	while(std::pow(2,d) < width)
	{
		d++;
	}


	width = std::pow(2,d);



	std::vector< std::vector< std::complex<double> > > f(width, std::vector< std::complex<double> >(width));



	for(int y=0;y<width;y++)
	{
		for(int x=0;x<width;x++)
		{
			if(x < filter_cols && y < filter_rows)
			{
				f[y][x] = std::complex<double> (filter[y][x], 0);
			}
			else
			{
				f[y][x] = std::complex<double> (0, 0);
			}
		}
	}

	std::vector< std::vector< std::complex<double> > > F1 = fft(f, true);




	clock_t start = clock();



	for(int y=0;y<width;y++)
	{
		for(int x=0;x<width;x++)
		{
			if(x < input.cols && y < input.rows)
			{
				f[y][x] = std::complex<double> (input.data[y * input.cols + x], 0);
			}
			else
			{
				f[y][x] = std::complex<double> (0, 0);
			}
		}
	}



	std::vector< std::vector< std::complex<double> > > F2 = fft(f, true);







	std::vector< std::vector< std::complex<double> > > F3(width, std::vector< std::complex<double> >(width));

	for(int y=0;y<width;y++)
	{
		for(int x=0;x<width;x++)
		{
			F3[y][x] = F1[y][x] * F2[y][x];
		}
	}


	std::vector< std::vector< std::complex<double> > > F = fft(F3, false);

	cv::Mat output = cv::Mat::zeros(width, width, CV_32FC1);


	for(int y=0;y<width;y++)
	{
		for(int x=0;x<width;x++)
		{
			output.at<float>(y, x) = F[y][x].real();
		}
	}



	std::cout << "convolution_fft (precomputed filter) : " << (double)clock()-start << std::endl;

	return output;

}







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