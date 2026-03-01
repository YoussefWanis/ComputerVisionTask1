#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

class Filters {
public:
    // Manual average (box) filter
    static cv::Mat average(const cv::Mat& input, int kernelSize = 3);

    // Manual Gaussian filter (separable)
    static cv::Mat gaussian(const cv::Mat& input, int kernelSize = 3, double sigma = 1.0);

    // Manual median filter
    static cv::Mat median(const cv::Mat& input, int kernelSize = 3);

private:
    Filters() = delete;
};

#endif // FILTERS_H