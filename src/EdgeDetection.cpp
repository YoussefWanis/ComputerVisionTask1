#include "EdgeDetection.h"
#include "Utils.h"
#include <cmath>

cv::Mat EdgeDetection::sobel(const cv::Mat& input) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat Gx = (cv::Mat_<float>(3,3) << -1,0,1, -2,0,2, -1,0,1);
    cv::Mat Gy = (cv::Mat_<float>(3,3) << -1,-2,-1, 0,0,0, 1,2,1);

    cv::Mat ix = Utils::convolve(gray, Gx);
    cv::Mat iy = Utils::convolve(gray, Gy);

    cv::Mat mag(gray.size(), CV_8U);
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            float gx = ix.at<uchar>(i,j);
            float gy = iy.at<uchar>(i,j);
            mag.at<uchar>(i,j) = cv::saturate_cast<uchar>(std::sqrt(gx*gx + gy*gy));
        }
    }
    return mag;
}

cv::Mat EdgeDetection::roberts(const cv::Mat& input) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat ix = cv::Mat::zeros(gray.size(), CV_32F);
    cv::Mat iy = cv::Mat::zeros(gray.size(), CV_32F);

    for (int i = 0; i < gray.rows-1; ++i) {
        for (int j = 0; j < gray.cols-1; ++j) {
            float gx = gray.at<uchar>(i,j)   * 1 + gray.at<uchar>(i+1,j+1) * (-1);
            float gy = gray.at<uchar>(i,j+1) * 1 + gray.at<uchar>(i+1,j)   * (-1);
            ix.at<float>(i,j) = gx;
            iy.at<float>(i,j) = gy;
        }
    }

    cv::Mat mag(gray.size(), CV_8U);
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            mag.at<uchar>(i,j) = cv::saturate_cast<uchar>(
                std::sqrt(ix.at<float>(i,j)*ix.at<float>(i,j) + iy.at<float>(i,j)*iy.at<float>(i,j)));
        }
    }
    return mag;
}

cv::Mat EdgeDetection::prewitt(const cv::Mat& input) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat Gx = (cv::Mat_<float>(3,3) << -1,0,1, -1,0,1, -1,0,1);
    cv::Mat Gy = (cv::Mat_<float>(3,3) << -1,-1,-1, 0,0,0, 1,1,1);

    cv::Mat ix = Utils::convolve(gray, Gx);
    cv::Mat iy = Utils::convolve(gray, Gy);

    cv::Mat mag(gray.size(), CV_8U);
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            float gx = ix.at<uchar>(i,j);
            float gy = iy.at<uchar>(i,j);
            mag.at<uchar>(i,j) = cv::saturate_cast<uchar>(std::sqrt(gx*gx + gy*gy));
        }
    }
    return mag;
}

cv::Mat EdgeDetection::canny(const cv::Mat& input, int lowThresh, int highThresh) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat edges;
    cv::Canny(gray, edges, lowThresh, highThresh);
    return edges;
}