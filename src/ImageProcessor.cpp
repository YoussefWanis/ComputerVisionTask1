#include "ImageProcessor.h"

QImage ImageProcessor::matToQImage(const cv::Mat& mat) {
    if (mat.empty()) return QImage();

    if (mat.type() == CV_8UC3) {
        cv::Mat temp;
        cv::cvtColor(mat, temp, cv::COLOR_BGR2RGB);
        return QImage((const uchar*)temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888).copy();
    } else if (mat.type() == CV_8UC1) {
        return QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    }
    return QImage();
}

cv::Mat ImageProcessor::toGrayscale(const cv::Mat& input) {
    if (input.channels() < 3) return input;
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat ImageProcessor::detectEdges(const cv::Mat& input, int lowThresh) {
    cv::Mat gray = toGrayscale(input);
    cv::Mat edges;
    cv::Canny(gray, edges, lowThresh, lowThresh * 3);
    return edges;
}

// Add implementations for Filter, Histogram, Frequency, etc. here