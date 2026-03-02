/**
 * @file ColorProcessor.cpp
 * @brief Implements colour-space conversion utilities.
 *
 * Provides manual (pixel-level) BGR-to-grayscale conversion using the
 * BT.601 luminance formula, and a thin wrapper around OpenCV's
 * cv::split for extracting individual colour channels.
 *
 * These helpers mirror the functionality found in the Python code-base
 * (Processors/color.py) and are used throughout the C++ pipeline
 * whenever grayscale or per-channel processing is required.
 */

#include "processors/ColorProcessor.h"

/**
 * @brief Convert a BGR colour image to single-channel grayscale.
 *
 * Uses the ITU-R BT.601 luminance formula:
 *     Y = 0.299 × R  +  0.587 × G  +  0.114 × B
 *
 * OpenCV stores colour images in BGR order, so the channel indices are:
 *     B = channel 0,  G = channel 1,  R = channel 2.
 *
 * If the input is already single-channel (grayscale), it is simply
 * cloned and returned unchanged.
 *
 * @param bgr  Input image in BGR format (CV_8UC3) or grayscale (CV_8UC1).
 * @return     A new single-channel CV_8UC1 grayscale image.
 */
cv::Mat ColorProcessor::toGrayscale(const cv::Mat& bgr) {
    // If already single-channel, clone and return as-is
    if (bgr.channels() == 1) return bgr.clone();

    // Allocate output matrix (same dimensions, single channel, 8-bit unsigned)
    cv::Mat gray(bgr.rows, bgr.cols, CV_8UC1);

    // Iterate over every pixel and apply the BT.601 luminance weights
    for (int i = 0; i < bgr.rows; ++i) {
        const uchar* src = bgr.ptr<uchar>(i);   // pointer to row i of the source
        uchar* dst = gray.ptr<uchar>(i);         // pointer to row i of the destination
        for (int j = 0; j < bgr.cols; ++j) {
            // Extract B, G, R from interleaved BGR layout
            double b = src[j * 3 + 0];
            double g = src[j * 3 + 1];
            double r = src[j * 3 + 2];
            // Weighted sum ? saturate-cast to [0, 255]
            dst[j] = cv::saturate_cast<uchar>(0.114 * b + 0.587 * g + 0.299 * r);
        }
    }
    return gray;
}

/**
 * @brief Split a BGR image into its three individual colour channels.
 *
 * Wraps OpenCV's cv::split to separate the interleaved 3-channel Mat
 * into a vector of single-channel Mats.
 *
 * @param bgr  Input BGR image (CV_8UC3).
 * @return     A vector of three CV_8UC1 Mats:
 *             - channels[0] ? Blue channel
 *             - channels[1] ? Green channel
 *             - channels[2] ? Red channel
 */
std::vector<cv::Mat> ColorProcessor::splitChannels(const cv::Mat& bgr) {
    std::vector<cv::Mat> channels;
    cv::split(bgr, channels);   // channels[0]=B, [1]=G, [2]=R
    return channels;
}
