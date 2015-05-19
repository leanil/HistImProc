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
		loc_hist[i] = 0.0f;
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

__global__ void histogram_equalizer_kernel(uchar* img, size_t pitch, int width, int height, float* map) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;
	for (int y = row; y < height; y += ny) {
		for (int x = col; x < width; x += nx) {
			uchar* pos = img + y*pitch + x;
			*pos = map[*pos];
		}
	}
}

__global__ void otsu_thresholding_kernel(uchar* img, size_t pitch, int width, int height, float t) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int nx = blockDim.x * gridDim.x;
	int ny = blockDim.y * gridDim.y;
	for (int y = row; y < height; y += ny) {
		for (int x = col; x < width; x += nx) {
			uchar* pos = img + y*pitch + x;
			*pos = (*pos>t) ? 255.0f : 0.0f;
		}
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

Mat calculate_histogram(const Mat& img) {
	uchar* d_img;
	size_t pitch;
	copy_image_to_device(img, d_img, pitch);

	const unsigned &width = img.size().width, &height = img.size().height;
	float* d_hist;
	check(cudaMalloc(&d_hist, 256 * sizeof(float)));
	check(cudaMemset(d_hist, 0, 256 * sizeof(float)));

	cudaEvent_t start, stop;
	check(cudaEventCreate(&start));
	check(cudaEventCreate(&stop));

	check(cudaEventRecord(start));
	histogram_kernel << <dim_grid, dim_block >> >(d_img, pitch, width, height, d_hist);
	check(cudaEventRecord(stop));

	Mat result(256, 1, CV_32FC1);
	check(cudaMemcpy(result.data, d_hist, result.total()*result.elemSize(), cudaMemcpyDeviceToHost));
	check(cudaEventSynchronize(stop));
	float milliseconds = 0;
	check(cudaEventElapsedTime(&milliseconds, start, stop));
	cout << "histogram calculation on gpu: " << milliseconds << " ms\n";
	check(cudaEventDestroy(start));
	check(cudaEventDestroy(stop));
	return result;
}

Mat equalize_histogram(const Mat& img) {
	Mat hist = calculate_histogram(img);
	for (int i = 1; i < hist.size[0]; i++) {
		hist.at<float>(i) += hist.at<float>(i - 1);
	}
	hist *= 255.0f / hist.at<float>(hist.size[0] - 1);

	uchar* d_img;
	size_t pitch;
	copy_image_to_device(img, d_img, pitch);

	const unsigned &width = img.size().width, &height = img.size().height;
	float* d_map;
	check(cudaMalloc(&d_map, hist.total() * hist.elemSize()));
	check(cudaMemcpy(d_map, hist.data, hist.total() * hist.elemSize(), cudaMemcpyHostToDevice));

	histogram_equalizer_kernel << <dim_grid, dim_block >> >(d_img, pitch, width, height, d_map);

	Mat result(height, width, img.type());
	check(cudaMemcpy2D(result.data, width, d_img, pitch, width * img.elemSize(), height, cudaMemcpyDeviceToHost));
	return result;
}

Mat otsu_thresholding(const cv::Mat& img) {
	Mat P = calculate_histogram(img);
	P /= sum(P)[0];
	float avg = 0;
	for (int i = 0; i < P.size[0]; i++) {
		avg += i*P.at<float>(i);
	}
	int t;
	float sub_sum = P.at<float>(0), sub_avg1 = 0, sub_avg2, t_var = 0;
	for (int i = 0; i < P.size[0] - 1; i++) {
		if (P.at<float>(i + 1) == 0) { continue; }
		sub_avg2 = (avg - sub_sum*sub_avg1) / (1 - sub_sum);
		float var = sub_sum*(1 - sub_sum)*(sub_avg1 - sub_avg2)*(sub_avg1 - sub_avg2);
		if (var > t_var) {
			t_var = var;
			t = i;
		}
		float old = sub_sum;
		sub_sum += P.at<float>(i + 1);
		sub_avg1 = (old*sub_avg1 + (i + 1)*P.at<float>(i + 1)) / (sub_sum);
	}

	uchar* d_img;
	size_t pitch;
	copy_image_to_device(img, d_img, pitch);
	const unsigned &width = img.size().width, &height = img.size().height;
	float ft = (float)t;
	otsu_thresholding_kernel << <dim_grid, dim_block >> >(d_img, pitch, width, height, ft);

	Mat result(height, width, img.type());
	check(cudaMemcpy2D(result.data, width, d_img, pitch, width * img.elemSize(), height, cudaMemcpyDeviceToHost));
	return result;
}