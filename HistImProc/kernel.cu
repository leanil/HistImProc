#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

const int BLOCK_SIZE = 16, GRID_SIZE = 8;
const dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE), dim_grid(GRID_SIZE, GRID_SIZE);
const int stride = 16;
template<class T>
inline T div_ceil(T a, T b) { return a / b + !(a % b); }

#ifdef _DEBUG
#define check(code) checkCuda(code, __FILE__, __LINE__)
#else
#define check(code) code
#endif

inline void checkCuda(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		cerr << "Cuda error: " << cudaGetErrorString(code) << " " << file << " " << line << endl;
		if (abort) { exit(code); }
	}
}

void copy_image_to_device(const Mat& img, uchar*& d_img, size_t& pitch) {
	const unsigned &width = img.size().width, &height = img.size().height;
	check(cudaMallocPitch(&d_img, &pitch, width * img.elemSize(), height));
	check(cudaMemcpy2D(d_img, pitch, img.data, width, width * img.elemSize(), height, cudaMemcpyHostToDevice));
}

__global__ void brightness_kernel(uchar* img, size_t pitch, int width, int height, int diff) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;
	for (int y = row; y < height; y += ny) {
		for (int x = col; x < width; x += nx) {
			uchar* pos = img + y*pitch + x;
			if (diff + *pos < 0) { *pos = 0; }
			else if (diff + *pos > 255) { *pos = 255; }
			else { *pos += diff; }
		}
	}
}

__global__ void histogram_kernel(uchar* img, size_t pitch, int width, int height, float* hist) {
	__shared__ float loc_hist[256];
	int col = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
	int row = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
	int nx = blockDim.x * gridDim.x * stride;
	int ny = blockDim.y * gridDim.y * stride;

	int linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
	int block_size = blockDim.x * blockDim.y;
	for (int i = linear_idx; i < 256; i += block_size) {
		loc_hist[i] = 0;
	}

	for (int y = row; y < height; y += ny) {
		for (int x = col; x < width; x += nx) {
			for (int dy = 0; dy < stride; ++dy) {
				for (int dx = 0; dx < stride; ++dx) {
					uchar* pos = img + (y + dy)*pitch + x + dx;
					atomicAdd(loc_hist + *pos, 1.0f);
				}
			}
		}
	}
	__syncthreads();
	for (int i = linear_idx; i < 256; i += block_size) {
		atomicAdd(hist + i, loc_hist[i]);
	}
}

Mat adjust_brightness(const Mat& img, int diff) {
	uchar* d_img;
	size_t pitch;
	copy_image_to_device(img, d_img, pitch);

	const unsigned &width = img.size().width, &height = img.size().height;
	brightness_kernel << <dim_grid, dim_block >> >(d_img, pitch, width, height, diff);

	Mat result(height, width, img.type());
	check(cudaMemcpy2D(result.data, width, d_img, pitch, width * img.elemSize(), height, cudaMemcpyDeviceToHost));
	return result;
}

Mat calculate_histogram(const cv::Mat& img) {
	uchar* d_img;
	size_t pitch;
	copy_image_to_device(img, d_img, pitch);

	const unsigned &width = img.size().width, &height = img.size().height;
	float* d_hist;
	check(cudaMalloc(&d_hist, 256 * sizeof(float)));
	check(cudaMemset(d_hist, 0.0f, 256 * sizeof(float)));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	histogram_kernel << <dim_grid, dim_block >> >(d_img, pitch, width, height, d_hist);
	cudaEventRecord(stop);

	Mat result(256, 1, CV_32FC1);
	check(cudaMemcpy(result.data, d_hist, result.total()*result.elemSize(), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "histogram calculation: " << milliseconds << " ms\n";
	return result;
}