#include "processors/FilterProcessor.h"
#include "processors/ColorProcessor.h"
#include "utils/ImageUtils.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

// ════════════════════════════════════════════════════════════════════
//  Public
// ════════════════════════════════════════════════════════════════════

cv::Mat FilterProcessor::process(const cv::Mat& image,
                                 const std::string& filterType,
                                 int kernelSize) {
    ImageUtils::assertNotEmpty(image, "FilterProcessor::process");
    ImageUtils::validateKernelSize(kernelSize);

    // Select the per-channel filter function
    using Fn = cv::Mat (FilterProcessor::*)(const cv::Mat&, int);
    Fn fn = nullptr;
    if      (filterType == "average")  fn = &FilterProcessor::averageFilter;
    else if (filterType == "gaussian") fn = &FilterProcessor::gaussianFilter;
    else if (filterType == "median")   fn = &FilterProcessor::medianFilter;
    else throw std::invalid_argument("Unknown filter_type: " + filterType);

    // Process each channel independently (matches Python behaviour)
    std::vector<cv::Mat> channels = ColorProcessor::splitChannels(image);
    for (auto& ch : channels)
        ch = (this->*fn)(ch, kernelSize);

    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

// ════════════════════════════════════════════════════════════════════
//  Gaussian — sigma = kernelSize / 6  (matches Python)
// ════════════════════════════════════════════════════════════════════

cv::Mat FilterProcessor::gaussianFilter(const cv::Mat& channel, int k) {
    double sigma = k / 6.0;
    int half = k / 2;

    // Build 1-D Gaussian kernel
    std::vector<double> g1d(k);
    double sum = 0;
    for (int i = 0; i < k; ++i) {
        double x = i - half;
        g1d[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += g1d[i];
    }
    for (auto& v : g1d) v /= sum;

    // Build 2-D kernel (outer product)
    cv::Mat kernel(k, k, CV_64F);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c)
            kernel.at<double>(r, c) = g1d[r] * g1d[c];

    // Delegate pad + convolve + clip to the shared utility
    return ImageUtils::applyKernelReflect(channel, kernel);
}

// ════════════════════════════════════════════════════════════════════
//  Average (box) filter
// ════════════════════════════════════════════════════════════════════

cv::Mat FilterProcessor::averageFilter(const cv::Mat& channel, int k) {
    int half = k / 2;
    cv::Mat padded = ImageUtils::padReflect(channel, half, half);

    cv::Mat output(channel.rows, channel.cols, CV_8UC1);
    double area = k * k;

    for (int i = 0; i < channel.rows; ++i) {
        for (int j = 0; j < channel.cols; ++j) {
            double sum = 0;
            for (int m = 0; m < k; ++m)
                for (int n = 0; n < k; ++n)
                    sum += padded.at<uchar>(i + m, j + n);
            output.at<uchar>(i, j) = static_cast<uchar>(sum / area);
        }
    }
    return output;
}

// ════════════════════════════════════════════════════════════════════
//  Median filter
// ════════════════════════════════════════════════════════════════════

cv::Mat FilterProcessor::medianFilter(const cv::Mat& channel, int k) {
    int half = k / 2;
    cv::Mat padded = ImageUtils::padReflect(channel, half, half);

    cv::Mat output(channel.rows, channel.cols, CV_8UC1);
    std::vector<uchar> window(k * k);

    for (int i = 0; i < channel.rows; ++i) {
        for (int j = 0; j < channel.cols; ++j) {
            int idx = 0;
            for (int m = 0; m < k; ++m)
                for (int n = 0; n < k; ++n)
                    window[idx++] = padded.at<uchar>(i + m, j + n);

            std::nth_element(window.begin(),
                             window.begin() + window.size() / 2,
                             window.end());
            output.at<uchar>(i, j) = window[window.size() / 2];
        }
    }
    return output;
}
