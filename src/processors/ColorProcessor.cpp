#include "processors/ColorProcessor.h"

cv::Mat ColorProcessor::toGrayscale(const cv::Mat& bgr) {
    if (bgr.channels() == 1) return bgr.clone();
    // BT.601 luminance: 0.299 R + 0.587 G + 0.114 B
    // OpenCV BGR order:  B=ch0, G=ch1, R=ch2
    cv::Mat gray(bgr.rows, bgr.cols, CV_8UC1);
    for (int i = 0; i < bgr.rows; ++i) {
        const uchar* src = bgr.ptr<uchar>(i);
        uchar* dst = gray.ptr<uchar>(i);
        for (int j = 0; j < bgr.cols; ++j) {
            double b = src[j * 3 + 0];
            double g = src[j * 3 + 1];
            double r = src[j * 3 + 2];
            dst[j] = cv::saturate_cast<uchar>(0.114 * b + 0.587 * g + 0.299 * r);
        }
    }
    return gray;
}

std::vector<cv::Mat> ColorProcessor::splitChannels(const cv::Mat& bgr) {
    std::vector<cv::Mat> channels;
    cv::split(bgr, channels);          // channels[0]=B, [1]=G, [2]=R
    return channels;
}
