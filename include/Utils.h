#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <QImage>

class Utils {
public:
    // Convert cv::Mat to QImage for display
    static QImage matToQImage(const cv::Mat& mat);

    // Convert color to grayscale (if needed)
    static cv::Mat toGrayscale(const cv::Mat& input);

    // Manual convolution (grayscale only)
    static cv::Mat convolve(const cv::Mat& input, const cv::Mat& kernel);

    // Generate 1D Gaussian kernel (float)
    static cv::Mat getGaussianKernel1D(int size, double sigma);

    // Compute histogram (grayscale or a specific channel)
    static cv::Mat computeHist(const cv::Mat& input, int channel = -1);

    // Compute cumulative histogram
    static cv::Mat computeCumulativeHist(const cv::Mat& hist);

    // Draw a histogram as an image
    static cv::Mat drawHistogram(const cv::Mat& hist, const cv::Scalar& color,
                                 int height = 300, int width = 512);

    // resize image keeping aspect ratio, max dimension = maxSize
    static cv::Mat resizeAspect(const cv::Mat& input, int maxSize = 512);
private:
    Utils() = delete; // static only
};

#endif // UTILS_H