#ifndef COLORPROCESSOR_H
#define COLORPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * ColorProcessor — colour-space helpers.
 * Mirrors Python's Processors/color.py.
 */
class ColorProcessor {
public:
    /** BGR → single-channel grayscale using BT.601 weights. */
    static cv::Mat toGrayscale(const cv::Mat& bgr);

    /** Split a BGR image into {B, G, R} single-channel Mats. */
    static std::vector<cv::Mat> splitChannels(const cv::Mat& bgr);

private:
    ColorProcessor() = delete;
};

#endif // COLORPROCESSOR_H
