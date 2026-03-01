#include "Threshold.h"
#include "Utils.h"

cv::Mat Threshold::global(const cv::Mat& input, int thresh) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat output(gray.size(), CV_8U);

    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            output.at<uchar>(i,j) = (gray.at<uchar>(i,j) > thresh) ? 255 : 0;
        }
    }
    return output;
}

cv::Mat Threshold::local(const cv::Mat& input, int blockSize, int constant) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat output = cv::Mat::zeros(gray.size(), CV_8U);
    int radius = blockSize / 2;

    for (int i = radius; i < gray.rows - radius; ++i) {
        for (int j = radius; j < gray.cols - radius; ++j) {
            int sum = 0;
            for (int m = -radius; m <= radius; ++m) {
                for (int n = -radius; n <= radius; ++n) {
                    sum += gray.at<uchar>(i+m, j+n);
                }
            }
            int mean = sum / (blockSize * blockSize);
            if (gray.at<uchar>(i,j) > mean - constant)
                output.at<uchar>(i,j) = 255;
        }
    }
    return output;
}