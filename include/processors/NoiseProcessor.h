#ifndef NOISEPROCESSOR_H
#define NOISEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <random>

/**
 * NoiseProcessor — adds noise to images.
 * Mirrors Python's Processors/noise.py.
 *
 * Supported types: "uniform", "gaussian", "salt_pepper".
 * All use an `intensity` factor in [0, 1].
 */
class NoiseProcessor {
public:
    explicit NoiseProcessor(unsigned int seed = 0);

    /**
     * @param image       Input image (BGR uint8).
     * @param noiseType   "uniform" | "gaussian" | "salt_pepper".
     * @param intensity   Noise strength in [0, 1].
     * @param mean        Gaussian mean  (gaussian only).
     * @param stddev      Gaussian base σ (gaussian only, scaled by intensity).
     * @param spRatio     Salt fraction   (salt_pepper only, in [0, 1]).
     */
    cv::Mat process(const cv::Mat& image,
                    const std::string& noiseType = "gaussian",
                    double intensity = 0.3,
                    double mean     = 0.0,
                    double stddev   = 100.0,
                    double spRatio  = 0.5);

private:
    cv::Mat addUniform   (const cv::Mat& image, double intensity);
    cv::Mat addGaussian  (const cv::Mat& image, double intensity,
                          double mean, double stddev);
    cv::Mat addSaltPepper(const cv::Mat& image, double intensity,
                          double ratio);

    std::mt19937 rng_;
};

#endif // NOISEPROCESSOR_H
