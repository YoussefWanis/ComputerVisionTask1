/**
 * @file FilterProcessor.h
 * @brief Declaration of the FilterProcessor class for spatial-domain
 *        image filtering.
 *
 * Mirrors the Python implementation in Processors/filter.py.
 * Supports average (box), Gaussian, and median filters applied
 * independently to each colour channel of a BGR image.
 */

#ifndef FILTERPROCESSOR_H
#define FILTERPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include "utils/ImageUtils.h"

/**
 * @class FilterProcessor
 * @brief Spatial-domain noise-reduction filters for single- or
 *        multi-channel images.
 *
 * Each filter operates per-channel: the input BGR image is split into
 * B, G, R planes, each plane is filtered independently, and the three
 * results are merged back.
 *
 * Supported filter types:
 *   - "average"  — unweighted box (mean) filter.
 *   - "gaussian" — Gaussian-weighted smoothing (σ = kernelSize / 6).
 *   - "median"   — non-linear median filter (good against salt & pepper).
 */
class FilterProcessor {
public:
    /** @brief Default constructor. */
    FilterProcessor() = default;

    /**
     * @brief Apply the selected spatial filter to the image.
     *
     * @param image      Input BGR image (CV_8UC3).
     * @param filterType Filter algorithm: "average" | "gaussian" | "median".
     * @param kernelSize Side length of the square kernel window.
     *                   Must be an odd integer ≥ 3.
     * @return           Filtered CV_8UC3 image of the same size.
     *
     * @throws std::invalid_argument  If the image is empty, the kernel
     *         size is invalid, or the filter type is unknown.
     */
    cv::Mat process(const cv::Mat& image,
                    const std::string& filterType,
                    int kernelSize);

private:
    /**
     * @brief Gaussian-weighted spatial smoothing on a single channel.
     * @param channel  Single-channel input (CV_8UC1).
     * @param kernelSize  Odd kernel side length (≥ 3).
     * @return  Smoothed CV_8UC1 channel.
     */
    cv::Mat gaussianFilter(const cv::Mat& channel, int kernelSize);

    /**
     * @brief Unweighted box (average / mean) filter on a single channel.
     * @param channel  Single-channel input (CV_8UC1).
     * @param kernelSize  Odd kernel side length (≥ 3).
     * @return  Smoothed CV_8UC1 channel.
     */
    cv::Mat averageFilter (const cv::Mat& channel, int kernelSize);

    /**
     * @brief Non-linear median filter on a single channel.
     * @param channel  Single-channel input (CV_8UC1).
     * @param kernelSize  Odd kernel side length (≥ 3).
     * @return  Filtered CV_8UC1 channel.
     */
    cv::Mat medianFilter  (const cv::Mat& channel, int kernelSize);
};

#endif // FILTERPROCESSOR_H
