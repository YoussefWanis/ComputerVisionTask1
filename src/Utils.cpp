#include "Utils.h"
#include <algorithm>
#include <cmath>

QImage Utils::matToQImage(const cv::Mat& mat) {
    if (mat.empty()) return QImage();
    if (mat.type() == CV_8UC3) {
        cv::Mat temp;
        cv::cvtColor(mat, temp, cv::COLOR_BGR2RGB);
        return QImage((const uchar*)temp.data, temp.cols, temp.rows,
                      temp.step, QImage::Format_RGB888).copy();
    } else if (mat.type() == CV_8UC1) {
        return QImage((const uchar*)mat.data, mat.cols, mat.rows,
                      mat.step, QImage::Format_Grayscale8).copy();
    }
    return QImage();
}

cv::Mat Utils::toGrayscale(const cv::Mat& input) {
    if (input.channels() == 1) return input;
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat Utils::resizeAspect(const cv::Mat& input, int maxSize) {
    if (input.cols <= maxSize && input.rows <= maxSize)
        return input.clone();
    double scale = std::min((double)maxSize / input.cols, (double)maxSize / input.rows);
    int newW = std::round(input.cols * scale);
    int newH = std::round(input.rows * scale);
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
    return resized;
}

cv::Mat Utils::convolve(const cv::Mat& input, const cv::Mat& kernel) {
    CV_Assert(input.channels() == 1);
    cv::Mat output = cv::Mat::zeros(input.size(), CV_8U);
    int kCenterX = kernel.cols / 2;
    int kCenterY = kernel.rows / 2;

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < kernel.rows; ++m) {
                int mm = kernel.rows - 1 - m; // flipped row
                for (int n = 0; n < kernel.cols; ++n) {
                    int nn = kernel.cols - 1 - n; // flipped col
                    int ii = i + (m - kCenterY);
                    int jj = j + (n - kCenterX);
                    if (ii >= 0 && ii < input.rows && jj >= 0 && jj < input.cols) {
                        sum += input.at<uchar>(ii, jj) * kernel.at<float>(mm, nn);
                    }
                }
            }
            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(sum);
        }
    }
    return output;
}

cv::Mat Utils::getGaussianKernel1D(int size, double sigma) {
    cv::Mat kernel(1, size, CV_32F);
    int center = size / 2;
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        int x = i - center;
        kernel.at<float>(0, i) = std::exp(-(x*x) / (2*sigma*sigma));
        sum += kernel.at<float>(0, i);
    }
    kernel /= sum;
    return kernel;
}

cv::Mat Utils::computeHist(const cv::Mat& input, int channel) {
    std::vector<cv::Mat> bgr;
    if (channel >= 0) {
        cv::split(input, bgr);
    }
    const cv::Mat& src = (channel >= 0) ? bgr[channel] : input;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    return hist;
}

cv::Mat Utils::computeCumulativeHist(const cv::Mat& hist) {
    cv::Mat cdf = hist.clone();
    for (int i = 1; i < cdf.rows; ++i) {
        cdf.at<float>(i) += cdf.at<float>(i-1);
    }
    return cdf;
}

cv::Mat Utils::drawHistogram(const cv::Mat& hist, const cv::Scalar& color,
                              int height, int width) {
    double maxVal;
    cv::minMaxLoc(hist, nullptr, &maxVal);
    cv::Mat histImage(height, width, CV_8UC3, cv::Scalar(255,255,255));
    int binWidth = cvRound((double)width / hist.rows);
    for (int i = 0; i < hist.rows; ++i) {
        float binVal = hist.at<float>(i);
        int barHeight = cvRound(binVal * height / maxVal);
        cv::rectangle(histImage,
                      cv::Point(i*binWidth, height-1),
                      cv::Point((i+1)*binWidth-1, height - barHeight),
                      color, cv::FILLED);
    }
    return histImage;
}