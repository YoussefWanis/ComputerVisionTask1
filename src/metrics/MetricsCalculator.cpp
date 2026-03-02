/**
 * @file MetricsCalculator.cpp
 * @brief Implements image-quality metrics: MSE, PSNR, and SNR.
 *
 * Mirrors the Python implementation in Metrics/Quality.py.
 * All metrics compare an original (reference) image to a processed
 * (potentially degraded) version.  Multi-channel images are handled
 * by computing the metric per-channel and averaging.
 */

#include "metrics/MetricsCalculator.h"
#include <cmath>
#include <limits>

/**
 * @brief Compute the Mean Squared Error between two images.
 *
 * MSE = (1/N) * sum( (original[i] - processed[i])^2 )
 * For multi-channel images the per-channel MSE values are averaged.
 *
 * @param original   Reference image.
 * @param processed  Processed image (same size and type as original).
 * @return           MSE value >= 0.  Zero means identical images.
 */
double MetricsCalculator::mse(const cv::Mat& original,
                              const cv::Mat& processed) const {
    // Both images must match in size and type
    CV_Assert(original.size() == processed.size());
    CV_Assert(original.type() == processed.type());

    // Convert to 64-bit float for precise arithmetic
    cv::Mat diff;
    original.convertTo(diff, CV_64F);
    cv::Mat proc64;
    processed.convertTo(proc64, CV_64F);

    // Squared per-pixel difference
    diff = diff - proc64;
    diff = diff.mul(diff);

    // cv::mean returns one value per channel
    cv::Scalar s = cv::mean(diff);

    // Average across all channels
    double sum = 0;
    for (int c = 0; c < diff.channels(); ++c)
        sum += s[c];
    return sum / diff.channels();
}

/**
 * @brief Compute Peak Signal-to-Noise Ratio in decibels.
 *
 * PSNR = 10 * log10(255^2 / MSE).
 * Returns +infinity when MSE is zero (identical images).
 *
 * @param original   Reference image.
 * @param processed  Processed image.
 * @return           PSNR in dB.
 */
double MetricsCalculator::psnr(const cv::Mat& original,
                               const cv::Mat& processed) const {
    double m = mse(original, processed);
    if (m == 0.0)
        return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10((255.0 * 255.0) / m);
}

/**
 * @brief Compute Signal-to-Noise Ratio in decibels.
 *
 * SNR = 10 * log10( E[signal^2] / E[noise^2] )
 * where noise = processed - original.
 * Returns +infinity when noise power is zero.
 *
 * @param original   Reference image.
 * @param processed  Processed image.
 * @return           SNR in dB.
 */
double MetricsCalculator::snr(const cv::Mat& original,
                              const cv::Mat& processed) const {
    CV_Assert(original.size() == processed.size());
    CV_Assert(original.type() == processed.type());

    cv::Mat orig64, proc64;
    original.convertTo(orig64, CV_64F);
    processed.convertTo(proc64, CV_64F);

    // Noise = processed - original
    cv::Mat noise = proc64 - orig64;

    // Signal and noise power (per-channel means of squared values)
    cv::Scalar sigPow = cv::mean(orig64.mul(orig64));
    cv::Scalar noiPow = cv::mean(noise.mul(noise));

    // Average across channels
    double sp = 0, np = 0;
    for (int c = 0; c < orig64.channels(); ++c) {
        sp += sigPow[c];
        np += noiPow[c];
    }
    sp /= orig64.channels();
    np /= orig64.channels();

    if (np == 0.0)
        return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(sp / np);
}

/**
 * @brief Compute all three metrics (MSE, PSNR, SNR) at once.
 *
 * @param original   Reference image.
 * @param processed  Processed image.
 * @return           Map with keys "MSE", "PSNR", "SNR".
 */
std::map<std::string, double>
MetricsCalculator::computeAll(const cv::Mat& original,
                              const cv::Mat& processed) const {
    return {
        {"MSE",  mse(original, processed)},
        {"PSNR", psnr(original, processed)},
        {"SNR",  snr(original, processed)}
    };
}
