#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <opencv2/opencv.hpp>

class Histogram {
public:
    // Return an image of the grayscale histogram (optional cumulative)
    static cv::Mat computeHistogramImage(const cv::Mat& input, bool cumulative = false);

    // Return an image with R,G,B histograms side by side (optional cumulative)
    static cv::Mat computeRGBHistograms(const cv::Mat& input, bool cumulative = false);

    // Manual histogram equalization
    static cv::Mat equalize(const cv::Mat& input);

    // Contrast stretching (normalization)
    static cv::Mat normalize(const cv::Mat& input, double newMin = 0, double newMax = 255);

private:
    Histogram() = delete;
};

#endif // HISTOGRAM_H