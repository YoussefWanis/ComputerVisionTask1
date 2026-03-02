/**
 * @file NoiseProcessor.cpp
 * @brief Implements synthetic noise generation for testing denoising
 *        filters: uniform, Gaussian, and salt-and-pepper noise.
 *
 * Mirrors the Python implementation in Processors/noise.py.
 * All noise types accept an `intensity` parameter in [0, 1] that
 * controls the overall noise strength.
 */

#include "processors/NoiseProcessor.h"
#include "utils/ImageUtils.h"
#include <stdexcept>

/**
 * @brief Construct the noise processor and initialise its RNG.
 *
 * @param seed  Random seed for the Mersenne Twister.  If 0 (default),
 *              a non-deterministic seed is obtained from std::random_device.
 */
NoiseProcessor::NoiseProcessor(unsigned int seed)
    : rng_(seed == 0 ? std::random_device{}() : seed) {}

/**
 * @brief Add the specified type of noise to the image.
 *
 * Routes to the appropriate private helper (addUniform, addGaussian,
 * or addSaltPepper) based on the noiseType string.
 *
 * @param image      Input image (CV_8UC1 or CV_8UC3). Must not be empty.
 * @param noiseType  Noise model: "uniform" | "gaussian" | "salt_pepper".
 * @param intensity  Noise strength in [0, 1].
 * @param mean       Gaussian mean (only used for "gaussian").
 * @param stddev     Gaussian base σ, scaled by intensity (only for "gaussian").
 * @param spRatio    Salt fraction (only for "salt_pepper").
 * @return           Noisy image (same type and size as input).
 *
 * @throws std::invalid_argument  If the image is empty or the noise
 *         type is unrecognised.
 */
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

/**
 * @brief Add uniform noise in [-a, a] where a = intensity × 255.
 *
 * Each pixel value has a random offset drawn from a continuous
 * uniform distribution.  The result is clipped to [0, 255] by
 * cv::saturate_cast during the final CV_8U conversion.
 *
 * @param image      Input image (CV_8UC1 or CV_8UC3).
 * @param intensity  Controls the noise amplitude; 1.0 → offsets up to ±255.
 * @return           Noisy image clipped to [0, 255].
 */
cv::Mat NoiseProcessor::addUniform(const cv::Mat& image, double intensity) {
    double a = intensity * 255.0;   // Half-range of the uniform distribution
    std::uniform_real_distribution<double> dist(-a, a);

    // Convert to float so we can add real-valued noise
    cv::Mat fimg;
    image.convertTo(fimg, CV_64F);

    // Add noise to every element (all channels, contiguous in memory)
    int totalElems = fimg.rows * fimg.cols * fimg.channels();
    double* ptr = reinterpret_cast<double*>(fimg.data);
    for (int k = 0; k < totalElems; ++k)
        ptr[k] += dist(rng_);

    // Clip to [0, 255] and convert back to 8-bit unsigned
    cv::Mat out;
    fimg.convertTo(out, CV_8U);
    return out;
}

// ── Gaussian noise ─────────────────────────────────────────────────

/**
 * @brief Add Gaussian noise N(mean, σ × intensity) to every pixel.
 *
 * The effective standard deviation is `stddev × intensity`, so
 * intensity acts as a scaling factor on the noise spread.
 *
 * @param image      Input image.
 * @param intensity  Scale factor applied to stddev.
 * @param mean       Centre of the Gaussian distribution.
 * @param stddev     Base standard deviation (before intensity scaling).
 * @return           Noisy image clipped to [0, 255].
 */
cv::Mat NoiseProcessor::addGaussian(const cv::Mat& image,
                                    double intensity,
                                    double mean,
                                    double stddev) {
    // Effective σ = stddev × intensity
    std::normal_distribution<double> dist(mean, stddev * intensity);

    // Convert to float for additive noise
    cv::Mat fimg;
    image.convertTo(fimg, CV_64F);

    // Add Gaussian-distributed noise to every element
    int totalElems = fimg.rows * fimg.cols * fimg.channels();
    double* ptr = reinterpret_cast<double*>(fimg.data);
    for (int k = 0; k < totalElems; ++k)
        ptr[k] += dist(rng_);

    // Clip and convert back to uint8
    cv::Mat out;
    fimg.convertTo(out, CV_8U);
    return out;
}

// ── Salt & pepper noise ────────────────────────────────────────────

/**
 * @brief Add salt-and-pepper (impulse) noise to the image.
 *
 * For each pixel, a random value r ∈ [0, 1) is drawn:
 *   - If r < intensity × ratio       → pixel becomes salt  (255, white).
 *   - If r > 1 − intensity × (1−ratio) → pixel becomes pepper (0, black).
 *   - Otherwise the pixel is left unchanged.
 *
 * @param image      Input image (CV_8UC1 or CV_8UC3).
 * @param intensity  Overall fraction of pixels to corrupt.
 * @param ratio      Fraction of corrupted pixels that become salt.
 *                   0.5 gives equal salt and pepper.
 * @return           Noisy image with impulse corruption.
 */
cv::Mat NoiseProcessor::addSaltPepper(const cv::Mat& image,
                                      double intensity,
                                      double ratio) {
    cv::Mat result = image.clone();
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    // Precompute thresholds
    double saltThresh   = intensity * ratio;           // r < this → salt
    double pepperThresh = intensity * (1.0 - ratio);   // r > (1 − this) → pepper

    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            double r = dist01(rng_);   // Random value per pixel

            if (r < saltThresh) {
                // Salt → set all channels to maximum (255 = white)
                if (result.channels() == 1) {
                    result.at<uchar>(i, j) = 255;
                } else {
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
                }
            } else if (r > 1.0 - pepperThresh) {
                // Pepper → set all channels to minimum (0 = black)
                if (result.channels() == 1) {
                    result.at<uchar>(i, j) = 0;
                } else {
                    result.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                }
            }
            // Otherwise: pixel remains unchanged
        }
    }
    return result;
}
