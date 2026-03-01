#include "Noise.h"
#include <random>

cv::Mat Noise::addUniform(const cv::Mat& input, double low, double high) {
    cv::Mat noise = cv::Mat::zeros(input.size(), CV_8UC3);
    cv::randu(noise, low, high);
    cv::Mat result;
    cv::add(input, noise, result);
    return result;
}

cv::Mat Noise::addGaussian(const cv::Mat& input, double mean, double stddev) {
    cv::Mat noise = cv::Mat::zeros(input.size(), CV_8UC3);
    cv::randn(noise, mean, stddev);
    cv::Mat result;
    cv::add(input, noise, result);
    return result;
}

cv::Mat Noise::addSaltPepper(const cv::Mat& input, double prob) {
    cv::Mat result = input.clone();
    int total = result.rows * result.cols * result.channels();
    int numSalt = static_cast<int>(total * prob / 2);
    int numPepper = numSalt;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> rowDist(0, result.rows - 1);
    std::uniform_int_distribution<> colDist(0, result.cols - 1);

    // Salt (white)
    for (int k = 0; k < numSalt; ++k) {
        int i = rowDist(gen);
        int j = colDist(gen);
        if (result.channels() == 1) {
            result.at<uchar>(i, j) = 255;
        } else {
            result.at<cv::Vec3b>(i, j)[0] = 255;
            result.at<cv::Vec3b>(i, j)[1] = 255;
            result.at<cv::Vec3b>(i, j)[2] = 255;
        }
    }

    // Pepper (black)
    for (int k = 0; k < numPepper; ++k) {
        int i = rowDist(gen);
        int j = colDist(gen);
        if (result.channels() == 1) {
            result.at<uchar>(i, j) = 0;
        } else {
            result.at<cv::Vec3b>(i, j)[0] = 0;
            result.at<cv::Vec3b>(i, j)[1] = 0;
            result.at<cv::Vec3b>(i, j)[2] = 0;
        }
    }
    return result;
}