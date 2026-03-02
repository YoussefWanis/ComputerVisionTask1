/**
 * @file ColorProcessor.h
 * @brief Declaration of the ColorProcessor static utility class
 *        for colour-space conversions and channel operations.
 *
 * Mirrors the Python implementation in Processors/color.py.
 * Provides manual BGR-to-grayscale conversion (BT.601 luminance)
 * and a thin wrapper around OpenCV's cv::split for extracting
 * individual B, G, R channels.
 */

#ifndef COLORPROCESSOR_H
#define COLORPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class ColorProcessor
 * @brief Pure-static colour-space helpers (BGR ↔ Grayscale, channel split).
 *
 * Cannot be instantiated — all methods are static.
 */
class ColorProcessor {
public:
    /**
     * @brief Convert a BGR colour image to single-channel grayscale
     *        using the BT.601 luminance weights.
     *
     * Y = 0.299 × R + 0.587 × G + 0.114 × B
     *
     * If the input is already single-channel, it is cloned as-is.
     *
     * @param bgr  Input image (CV_8UC3 or CV_8UC1).
     * @return     New CV_8UC1 grayscale image.
     */
    static cv::Mat toGrayscale(const cv::Mat& bgr);

    /**
     * @brief Split a BGR image into three single-channel Mats.
     *
     * @param bgr  Input BGR image (CV_8UC3).
     * @return     Vector of 3 Mats: [0] = Blue, [1] = Green, [2] = Red.
     */
    static std::vector<cv::Mat> splitChannels(const cv::Mat& bgr);

private:
    ColorProcessor() = delete;   ///< Pure-static class — cannot be instantiated.
};

#endif // COLORPROCESSOR_H
