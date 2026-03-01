#ifndef EDGEDETECTION_H
#define EDGEDETECTION_H

#include <opencv2/opencv.hpp>

class EdgeDetection {
public:
    // Manual Sobel edge magnitude
    static cv::Mat sobel(const cv::Mat& input);

    // Manual Roberts edge magnitude
    static cv::Mat roberts(const cv::Mat& input);

    // Manual Prewitt edge magnitude
    static cv::Mat prewitt(const cv::Mat& input);

    // Canny (OpenCV allowed)
    static cv::Mat canny(const cv::Mat& input, int lowThresh = 100, int highThresh = 200);

private:
    EdgeDetection() = delete;
};

#endif // EDGEDETECTION_H