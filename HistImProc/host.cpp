#include "kernel.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

int main() {
	try {
		Mat src = imread("Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);
		//Mat result = adjust_brightness(src, 100);
		Mat result = equalize_histogram(src);
		imshow("original", src);
		imshow("result", result);

		int histSize = 256;
		float range[] { 0, 256 };
		const float* histRange{ range };
		Mat cpu_hist, gpu_hist;
		auto start = chrono::high_resolution_clock::now();
		calcHist(&src, 1, 0, Mat(), cpu_hist, 1, &histSize, &histRange);
		auto stop = chrono::high_resolution_clock::now();
		cout << "histogram calculation on cpu: "
			<< chrono::duration_cast<chrono::milliseconds>(stop - start).count() << " ms\n";
		gpu_hist = calculate_histogram(src);
		cout << equal(cpu_hist.begin<float>(), cpu_hist.end<float>(), gpu_hist.begin<float>()) << endl;
		waitKey();
	}
	catch (const cv::Exception& ex) {
		std::cout << "Error: " << ex.what() << std::endl;
	}

	return 0;
}