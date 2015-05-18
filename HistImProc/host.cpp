#include "kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

int main() {
	try {
		Mat src = imread("Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);
		Mat bright = adjust_brightness(src, 50);
		imshow("original", src);
		imshow("result", bright);

		int histSize = 256;
		float range[] { 0, 256 };
		const float* histRange{ range };
		Mat cpu_hist, gpu_hist;
		calcHist(&src, 1, 0, Mat(), cpu_hist, 1, &histSize, &histRange);
		gpu_hist = calculate_histogram(src);
		waitKey();
	}
	catch (const cv::Exception& ex) {
		std::cout << "Error: " << ex.what() << std::endl;
	}

	return 0;
}