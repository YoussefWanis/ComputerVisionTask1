/**
 * @file FFTProcessor.h
 * @brief Declaration of the FFTProcessor class for frequency-domain
 *        image filtering using ideal circular masks.
 *
 * Mirrors the Python implementation in Processors/fft.py.
 *
 * Provides both normalised (uint8) output for display and raw
 * (float64) output for arithmetic combination in hybrid images.
 */

#ifndef FFTPROCESSOR_H
#define FFTPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include "utils/ImageUtils.h"

/**
 * @class FFTProcessor
 * @brief Ideal circular low-pass / high-pass filter in the frequency domain.
 *
 * Pipeline per channel:
 *   1. Forward DFT.
 *   2. fftShift (DC to centre).
 *   3. Element-wise multiply with circular binary mask.
 *   4. ifftShift (undo shift).
 *   5. Inverse DFT.
 *   6. Take magnitude (abs of complex result).
 *
 * Multi-channel (BGR) images are split, filtered per-channel, and merged.
 */
class FFTProcessor {
public:
    /** @brief Default constructor. */
    FFTProcessor() = default;

    /**
     * @brief Filter an image and return normalised uint8 for display.
     *
     * The raw magnitude is min-max normalised to [0, 255].
     *
     * @param image        Input image (CV_8UC1 or CV_8UC3).
     * @param filterType   "lowpass" or "highpass".
     * @param cutoffRadius Radius (pixels) of the circular frequency mask.
     * @return             CV_8U image (single- or 3-channel).
     *
     * @throws std::invalid_argument  If the image is empty.
     */
    cv::Mat process(const cv::Mat& image,
                    const std::string& filterType,
                    int cutoffRadius);

    /**
     * @brief Filter an image and return raw float64 (unnormalised).
     *
     * Intended for arithmetic combination (e.g. hybrid images) where
     * the true magnitude values must be preserved.
     *
     * @param image        Input image (CV_8UC1 or CV_8UC3).
     * @param filterType   "lowpass" or "highpass".
     * @param cutoffRadius Radius (pixels) of the circular frequency mask.
     * @return             CV_64F image (single- or 3-channel).
     *
     * @throws std::invalid_argument  If the image is empty.
     */
    cv::Mat processRaw(const cv::Mat& image,
                       const std::string& filterType,
                       int cutoffRadius);

private:
    /**
     * @brief Build a binary circular frequency-domain mask.
     * @param shape         Spatial dimensions (rows × cols).
     * @param filterType    "lowpass" or "highpass".
     * @param cutoffRadius  Circle radius in pixels.
     * @return              CV_64F mask (0.0 or 1.0 per element).
     */
    cv::Mat makeCircularMask(cv::Size shape,
                             const std::string& filterType,
                             int cutoffRadius);

    /**
     * @brief Full DFT → mask → IDFT pipeline on one channel (raw output).
     * @param channel       Single-channel input (CV_8UC1).
     * @param filterType    "lowpass" or "highpass".
     * @param cutoffRadius  Circle radius in pixels.
     * @return              CV_64F filtered channel (unnormalised).
     */
    cv::Mat filterChannelRaw(const cv::Mat& channel,
                             const std::string& filterType,
                             int cutoffRadius);

    /**
     * @brief Full DFT → mask → IDFT pipeline on one channel (normalised).
     * @param channel       Single-channel input (CV_8UC1).
     * @param filterType    "lowpass" or "highpass".
     * @param cutoffRadius  Circle radius in pixels.
     * @return              CV_8UC1 filtered channel, [0, 255].
     */
    cv::Mat filterChannel(const cv::Mat& channel,
                          const std::string& filterType,
                          int cutoffRadius);
};

#endif // FFTPROCESSOR_H
