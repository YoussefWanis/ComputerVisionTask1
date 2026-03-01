#ifndef HYBRID_H
#define HYBRID_H

#include <opencv2/opencv.hpp>

class Hybrid {
public:
    // Create a hybrid image from two images (uses frequency domain filters)
    static cv::Mat create(const cv::Mat& img1, const cv::Mat& img2,
                          float cutoff1 = 30, float cutoff2 = 30);

private:
    Hybrid() = delete;
};

#endif // HYBRID_H