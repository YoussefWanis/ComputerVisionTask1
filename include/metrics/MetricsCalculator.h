/**
 * @file MetricsCalculator.h
 * @brief Declaration of the MetricsCalculator class for image-quality
 *        assessment.
 *
 * Mirrors the Python implementation in Metrics/Quality.py.
 * Computes three standard distortion metrics between an original
 * and a processed (potentially degraded) image:
 *   - MSE  (Mean Squared Error)
 *   - PSNR (Peak Signal-to-Noise Ratio)
 *   - SNR  (Signal-to-Noise Ratio)
 */

#ifndef METRICSCALCULATOR_H
#define METRICSCALCULATOR_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <cmath>

/**
 * @class MetricsCalculator
 * @brief Computes image-quality metrics comparing an original image
 *        to a processed version.
 *
 * All methods assume the two images have the same size and type.
 * Multi-channel images are handled by averaging the metric across
 * all channels.
 */
class MetricsCalculator {
public:
    /** @brief Default constructor. */
    MetricsCalculator() = default;

    /**
     * @brief Compute the Mean Squared Error (MSE).
     *
     * MSE = (1/N) × Σ (original[i] − processed[i])²
     *
     * Lower is better; MSE == 0 means the images are identical.
     *
     * @param original   Reference image.
     * @param processed  Processed / degraded image.
     * @return           MSE value (≥ 0).
     */
    double mse (const cv::Mat& original, const cv::Mat& processed) const;

    /**
     * @brief Compute the Peak Signal-to-Noise Ratio (PSNR) in decibels.
     *
     * PSNR = 10 × log10(255² / MSE)
     *
     * Higher is better; PSNR == ∞ when MSE == 0 (identical images).
     * Typical values for 8-bit images: 20–50 dB.
     *
     * @param original   Reference image.
     * @param processed  Processed / degraded image.
     * @return           PSNR in dB (or +∞ if MSE is zero).
     */
    double psnr(const cv::Mat& original, const cv::Mat& processed) const;

    /**
     * @brief Compute the Signal-to-Noise Ratio (SNR) in decibels.
     *
     * SNR = 10 × log10( E[original²] / E[(processed − original)²] )
     *
     * Measures how much signal power there is relative to noise power.
     * Higher is better; SNR == ∞ when the noise is zero.
     *
     * @param original   Reference image.
     * @param processed  Processed / degraded image.
     * @return           SNR in dB (or +∞ if noise power is zero).
     */
    double snr (const cv::Mat& original, const cv::Mat& processed) const;

    /**
     * @brief Compute all three metrics at once.
     *
     * Convenience method that returns a map with keys:
     *   "MSE", "PSNR", "SNR".
     *
     * @param original   Reference image.
     * @param processed  Processed / degraded image.
     * @return           Map of metric name → value.
     */
    std::map<std::string, double>
    computeAll(const cv::Mat& original, const cv::Mat& processed) const;
};

#endif // METRICSCALCULATOR_H
