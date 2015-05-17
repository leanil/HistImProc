#include "kernel.h"
#include <cstdio>

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaarithm.hpp"

int main() {
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	try {
		cv::Mat src_host = cv::imread("Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);
		cv::cuda::GpuMat dst, src;
		src.upload(src_host);

		cv::cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);

		cv::Mat result_host{ dst };
		cv::imshow("Result", result_host);
		cv::waitKey();
	}
	catch (const cv::Exception& ex) {
		std::cout << "Error: " << ex.what() << std::endl;
	}

	return 0;
}