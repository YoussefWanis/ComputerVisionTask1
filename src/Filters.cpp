#include "Filters.h"
#include "Utils.h"
#include <algorithm>

cv::Mat Filters::average(const cv::Mat& input, int kernelSize) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);
    return Utils::convolve(gray, kernel);
}

cv::Mat Filters::gaussian(const cv::Mat& input, int kernelSize, double sigma) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat kernel1D = Utils::getGaussianKernel1D(kernelSize, sigma);
    cv::Mat kernel2D = kernel1D.t() * kernel1D;
    return Utils::convolve(gray, kernel2D);
}

cv::Mat Filters::median(const cv::Mat& input, int kernelSize) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat output = gray.clone();
    int radius = kernelSize / 2;
    std::vector<uchar> neighbors;
    neighbors.reserve(kernelSize * kernelSize);

    for (int i = radius; i < gray.rows - radius; ++i) {
        for (int j = radius; j < gray.cols - radius; ++j) {
            neighbors.clear();
            for (int m = -radius; m <= radius; ++m) {
                for (int n = -radius; n <= radius; ++n) {
                    neighbors.push_back(gray.at<uchar>(i+m, j+n));
                }
            }
            std::nth_element(neighbors.begin(), neighbors.begin() + neighbors.size()/2, neighbors.end());
            output.at<uchar>(i, j) = neighbors[neighbors.size()/2];
        }
    }
    return output;
}