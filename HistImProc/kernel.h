#pragma once

#include <opencv2/opencv.hpp>

cv::Mat adjust_brightness(const cv::Mat& img, int diff);

cv::Mat calculate_histogram(const cv::Mat& img);

cv::Mat equalize_histogram(const cv::Mat& img);