#include "processors/NoiseProcessor.h"
#include "utils/ImageUtils.h"
#include <stdexcept>

NoiseProcessor::NoiseProcessor(unsigned int seed)
    : rng_(seed == 0 ? std::random_device{}() : seed) {}

cv::Mat NoiseProcessor::process(const cv::Mat& image,
                                const std::string& noiseType,
                                double intensity,
                                double mean,
                                double stddev,
                                double spRatio) {
    ImageUtils::assertNotEmpty(image, "NoiseProcessor::process");
    if (noiseType == "uniform")
        return addUniform(image, intensity);
    if (noiseType == "gaussian")
        return addGaussian(image, intensity, mean, stddev);
    if (noiseType == "salt_pepper")
        return addSaltPepper(image, intensity, spRatio);
    throw std::invalid_argument("Unknown noise_type: " + noiseType);
}

// ── Uniform noise ──────────────────────────────────────────────────
// Python: a = intensity * 255;  noise ~ U(-a, a)
cv::Mat NoiseProcessor::addUniform(const cv::Mat& image, double intensity) {
    double a = intensity * 255.0;
    std::uniform_real_distribution<double> dist(-a, a);

    cv::Mat fimg;
    image.convertTo(fimg, CV_64F);

    int totalElems = fimg.rows * fimg.cols * fimg.channels();
    double* ptr = reinterpret_cast<double*>(fimg.data);
    for (int k = 0; k < totalElems; ++k)
        ptr[k] += dist(rng_);

    cv::Mat out;
    fimg.convertTo(out, CV_8U);          // saturate_cast clips to [0,255]
    return out;
}

// ── Gaussian noise ─────────────────────────────────────────────────
// Python: noise ~ N(mean, std * intensity)
cv::Mat NoiseProcessor::addGaussian(const cv::Mat& image,
                                    double intensity,
                                    double mean,
                                    double stddev) {
    std::normal_distribution<double> dist(mean, stddev * intensity);

    cv::Mat fimg;
    image.convertTo(fimg, CV_64F);

    int totalElems = fimg.rows * fimg.cols * fimg.channels();
    double* ptr = reinterpret_cast<double*>(fimg.data);
    for (int k = 0; k < totalElems; ++k)
        ptr[k] += dist(rng_);

    cv::Mat out;
    fimg.convertTo(out, CV_8U);
    return out;
}

// ── Salt & pepper noise ────────────────────────────────────────────
// Python: per-pixel mask, salt if < intensity*ratio, pepper if > 1-intensity*(1-ratio)
cv::Mat NoiseProcessor::addSaltPepper(const cv::Mat& image,
                                      double intensity,
                                      double ratio) {
    cv::Mat result = image.clone();
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    double saltThresh   = intensity * ratio;
    double pepperThresh = intensity * (1.0 - ratio);

    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            double r = dist01(rng_);
            if (r < saltThresh) {
                // salt → 255 on every channel
                if (result.channels() == 1) {
                    result.at<uchar>(i, j) = 255;
                } else {
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
                }
            } else if (r > 1.0 - pepperThresh) {
                // pepper → 0
                if (result.channels() == 1) {
                    result.at<uchar>(i, j) = 0;
                } else {
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                }
            }
        }
    }
    return result;
}
