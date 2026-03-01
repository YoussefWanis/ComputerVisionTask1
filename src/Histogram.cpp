#include "Histogram.h"
#include "Utils.h"

cv::Mat Histogram::computeHistogramImage(const cv::Mat& input, bool cumulative) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat hist = Utils::computeHist(gray);
    if (cumulative) hist = Utils::computeCumulativeHist(hist);
    return Utils::drawHistogram(hist, cv::Scalar(0,0,0));
}

cv::Mat Histogram::computeRGBHistograms(const cv::Mat& input, bool cumulative) {
    CV_Assert(input.channels() == 3);

    std::vector<cv::Mat> bgr;
    cv::split(input, bgr);

    cv::Mat histR = Utils::computeHist(bgr[2]);
    cv::Mat histG = Utils::computeHist(bgr[1]);
    cv::Mat histB = Utils::computeHist(bgr[0]);

    if (cumulative) {
        histR = Utils::computeCumulativeHist(histR);
        histG = Utils::computeCumulativeHist(histG);
        histB = Utils::computeCumulativeHist(histB);
    }

    int height = 300, width = 512;
    cv::Mat canvas(height, width*3, CV_8UC3, cv::Scalar(255,255,255));

    cv::Mat rPlot = Utils::drawHistogram(histR, cv::Scalar(0,0,255), height, width);
    cv::Mat gPlot = Utils::drawHistogram(histG, cv::Scalar(0,255,0), height, width);
    cv::Mat bPlot = Utils::drawHistogram(histB, cv::Scalar(255,0,0), height, width);

    rPlot.copyTo(canvas(cv::Rect(0,0,width,height)));
    gPlot.copyTo(canvas(cv::Rect(width,0,width,height)));
    bPlot.copyTo(canvas(cv::Rect(2*width,0,width,height)));

    return canvas;
}

cv::Mat Histogram::equalize(const cv::Mat& input) {
    cv::Mat gray = Utils::toGrayscale(input);
    cv::Mat hist = Utils::computeHist(gray);
    cv::Mat cdf = Utils::computeCumulativeHist(hist);
    cdf /= gray.total();

    uchar lut[256];
    for (int i = 0; i < 256; ++i) {
        lut[i] = cv::saturate_cast<uchar>(255 * cdf.at<float>(i));
    }

    cv::Mat output(gray.size(), CV_8U);
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            output.at<uchar>(i,j) = lut[gray.at<uchar>(i,j)];
        }
    }
    return output;
}

cv::Mat Histogram::normalize(const cv::Mat& input, double newMin, double newMax) {
    cv::Mat gray = Utils::toGrayscale(input);
    double minVal, maxVal;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    cv::Mat output;
    gray.convertTo(output, CV_8U, (newMax - newMin) / (maxVal - minVal),
                   -minVal * (newMax - newMin) / (maxVal - minVal) + newMin);
    return output;
}