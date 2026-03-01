#ifndef NOISE_H
#define NOISE_H

#include <opencv2/opencv.hpp>

class Noise {
public:
    // Add uniform noise (OpenCV)
    static cv::Mat addUniform(const cv::Mat& input, double low = 0, double high = 50);

    // Add Gaussian noise (OpenCV)
    static cv::Mat addGaussian(const cv::Mat& input, double mean = 0, double stddev = 25);

    // Add salt & pepper noise (manual)
    static cv::Mat addSaltPepper(const cv::Mat& input, double prob = 0.05);

private:
    Noise() = delete;
};

#endif // NOISE_H