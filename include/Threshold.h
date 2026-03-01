#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <opencv2/opencv.hpp>

class Threshold {
public:
    // Global thresholding (manual)
    static cv::Mat global(const cv::Mat& input, int thresh = 128);

    // Local thresholding using mean of block (manual)
    static cv::Mat local(const cv::Mat& input, int blockSize = 11, int constant = 2);

private:
    Threshold() = delete;
};

#endif // THRESHOLD_H