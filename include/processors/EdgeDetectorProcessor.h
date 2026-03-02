#ifndef EDGEDETECTORPROCESSOR_H
#define EDGEDETECTORPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include "utils/ImageUtils.h"

/**
 * EdgeDetectorProcessor — gradient-based edge detection + Canny.
 * Mirrors Python's Processors/edge.py.
 *
 * Methods   : "sobel", "roberts", "prewitt", "canny".
 * Directions: "x", "y", "combined"  (gradient methods only).
 */
class EdgeDetectorProcessor {
public:
    EdgeDetectorProcessor() = default;

    cv::Mat process(const cv::Mat& image,
                    const std::string& method    = "sobel",
                    const std::string& direction = "combined",
                    int cannyLow  = 50,
                    int cannyHigh = 150);

private:
    cv::Mat detect     (const cv::Mat& image, const std::string& method,
                        const std::string& direction);
    cv::Mat cannyDetect(const cv::Mat& image, int low, int high);
};

#endif // EDGEDETECTORPROCESSOR_H
