#ifndef FILTERPROCESSOR_H
#define FILTERPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include "utils/ImageUtils.h"

/**
 * FilterProcessor — spatial-domain filtering, per-channel for colour images.
 * Mirrors Python's Processors/filter.py.
 *
 * Supported types: "average", "gaussian", "median".
 */
class FilterProcessor {
public:
    FilterProcessor() = default;

    /**
     * @param image      BGR uint8 input.
     * @param filterType "average" | "gaussian" | "median".
     * @param kernelSize Odd integer ≥ 3.
     */
    cv::Mat process(const cv::Mat& image,
                    const std::string& filterType,
                    int kernelSize);

private:
    cv::Mat gaussianFilter(const cv::Mat& channel, int kernelSize);
    cv::Mat averageFilter (const cv::Mat& channel, int kernelSize);
    cv::Mat medianFilter  (const cv::Mat& channel, int kernelSize);
};

#endif // FILTERPROCESSOR_H
