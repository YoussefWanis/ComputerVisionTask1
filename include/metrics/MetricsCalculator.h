#ifndef METRICSCALCULATOR_H
#define METRICSCALCULATOR_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <cmath>

/**
 * MetricsCalculator — image-quality metrics (MSE, PSNR, SNR).
 * Mirrors Python's Metrics/Quality.py.
 */
class MetricsCalculator {
public:
    MetricsCalculator() = default;

    double mse (const cv::Mat& original, const cv::Mat& processed) const;
    double psnr(const cv::Mat& original, const cv::Mat& processed) const;
    double snr (const cv::Mat& original, const cv::Mat& processed) const;

    std::map<std::string, double>
    computeAll(const cv::Mat& original, const cv::Mat& processed) const;
};

#endif // METRICSCALCULATOR_H
