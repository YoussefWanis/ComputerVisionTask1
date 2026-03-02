/**
 * @file NoiseProcessor.h
 * @brief Declaration of the NoiseProcessor class for adding synthetic
 *        noise to images.
 *
 * Mirrors the Python implementation in Processors/noise.py.
 * Supports uniform, Gaussian, and salt-and-pepper noise generation
 * with configurable intensity and distribution parameters.
 */

#ifndef NOISEPROCESSOR_H
#define NOISEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <random>

/**
 * @class NoiseProcessor
 * @brief Adds synthetic noise to images for testing denoising filters.
 *
 * Supported noise types:
 *   - **"uniform"**     — additive uniform noise in [-a, a] where
 *                         a = intensity × 255.
 *   - **"gaussian"**    — additive Gaussian noise N(mean, σ × intensity).
 *   - **"salt_pepper"** — per-pixel random replacement with white (salt)
 *                         or black (pepper) pixels.
 *
 * All noise types accept an `intensity` parameter in [0, 1] that
 * controls the overall noise strength.
 *
 * The internal Mersenne Twister RNG can be seeded for reproducibility.
 */
class NoiseProcessor {
public:
    /**
     * @brief Construct a NoiseProcessor with an optional RNG seed.
     *
     * @param seed  Random seed. If 0 (default), a non-deterministic
     *              seed is drawn from std::random_device.
     */
    explicit NoiseProcessor(unsigned int seed = 0);

    /**
     * @brief Add noise of the specified type to the image.
     *
     * @param image       Input image (CV_8UC1 or CV_8UC3).
     * @param noiseType   Noise model: "uniform" | "gaussian" | "salt_pepper".
     * @param intensity   Noise strength in [0, 1].  Higher values
     *                    produce more visible noise.
     * @param mean        Mean of the Gaussian distribution (used only
     *                    when noiseType == "gaussian").
     * @param stddev      Base standard deviation of the Gaussian noise
     *                    (scaled by intensity).  Used only for "gaussian".
     * @param spRatio     Fraction of noisy pixels that become salt (white)
     *                    vs. pepper (black).  Used only for "salt_pepper".
     *                    0.5 gives equal salt and pepper.
     * @return            Noisy image (same type and size as input).
     *
     * @throws std::invalid_argument  If the image is empty or an
     *         unknown noise type is specified.
     */
    cv::Mat process(const cv::Mat& image,
                    const std::string& noiseType = "gaussian",
                    double intensity = 0.3,
                    double mean     = 0.0,
                    double stddev   = 100.0,
                    double spRatio  = 0.5);

private:
    /**
     * @brief Add uniform noise U(-a, a) where a = intensity × 255.
     * @param image      Input image (CV_8UC1 or CV_8UC3).
     * @param intensity  Noise strength in [0, 1].
     * @return           Noisy image clipped to [0, 255].
     */
    cv::Mat addUniform(const cv::Mat& image, double intensity);

    /**
     * @brief Add Gaussian noise N(mean, stddev × intensity).
     * @param image      Input image.
     * @param intensity  Noise strength scale factor.
     * @param mean       Distribution mean.
     * @param stddev     Base standard deviation (multiplied by intensity).
     * @return           Noisy image clipped to [0, 255].
     */
    cv::Mat addGaussian(const cv::Mat& image, double intensity,
                        double mean, double stddev);

    /**
     * @brief Add salt-and-pepper (impulse) noise.
     * @param image      Input image.
     * @param intensity  Probability that any given pixel is corrupted.
     * @param ratio      Fraction of corrupted pixels that become salt
     *                   (white); the rest become pepper (black).
     * @return           Noisy image.
     */
    cv::Mat addSaltPepper(const cv::Mat& image, double intensity,
                          double ratio);

    std::mt19937 rng_;   ///< Mersenne Twister pseudo-random number generator.
};

#endif // NOISEPROCESSOR_H
