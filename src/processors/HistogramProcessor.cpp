/**
 * @file HistogramProcessor.cpp
 * @brief Implements histogram equalisation, normalisation, and
 *        per-channel histogram/CDF computation.
 *
 * Mirrors the Python implementation in Processors/histogram.py.
 *
 * Key algorithms:
 *   - **Single-channel equalisation** — classic CDF-based lookup-table
 *     mapping that spreads the intensity distribution evenly.
 *   - **BGR equalisation** — equalises the luminance channel and scales
 *     each colour component proportionally to preserve hue.
 *   - **Normalisation** — simple min-max linear stretch to [0, 255].
 *   - **Channel histograms** — computes 256-bin histograms and normalised
 *     CDFs for B, G, R independently.
 */

#include "processors/HistogramProcessor.h"
#include "processors/ColorProcessor.h"
#include "utils/ImageUtils.h"
#include <cmath>
#include <algorithm>

// ════════════════════════════════════════════════════════════════════
//  Public — equalize
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Equalise the histogram of a grayscale or BGR image.
 *
 * For single-channel images, the standard CDF-mapping equalisation is
 * applied directly.  For 3-channel BGR images, luminance-based
 * equalisation is used so colour ratios are preserved.
 *
 * @param image  Input image (CV_8UC1 or CV_8UC3). Must not be empty.
 * @return       Equalised image of the same type and size.
 *
 * @throws std::invalid_argument  If the image is empty.
 */
cv::Mat HistogramProcessor::equalize(const cv::Mat& image) {
    ImageUtils::assertNotEmpty(image, "HistogramProcessor::equalize");
    if (image.channels() == 1)
        return equalizeChannel(image);   // Direct grayscale path
    return equalizeBGR(image);           // Luminance-preserving colour path
}

// ════════════════════════════════════════════════════════════════════
//  Public — normalize  [min-max → 0..255]
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Normalise the image intensity range to [0, 255] using
 *        min-max linear scaling.
 *
 * All channels are flattened to find a single global minimum and
 * maximum, then the formula
 *     out = (pixel − min) / (max − min) × 255
 * is applied element-wise.
 *
 * @param image  Input image (any 8-bit type).
 * @return       Normalised CV_8U image.
 */
cv::Mat HistogramProcessor::normalize(const cv::Mat& image) {
    // Find global min / max across all channels
    double mn, mx;
    cv::minMaxLoc(image.reshape(1), &mn, &mx);   // reshape(1) flattens channels
    if (mn == mx) return image.clone();           // Constant image — nothing to stretch

    // Convert to float, apply linear mapping, convert back to uint8
    cv::Mat fimg;
    image.convertTo(fimg, CV_64F);
    fimg = (fimg - mn) / (mx - mn) * 255.0;

    cv::Mat out;
    fimg.convertTo(out, CV_8U);
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  Public — computeChannelHistograms
//
//  Returns keys "B", "G", "R" (adapted to BGR channel order).
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Compute 256-bin histograms and normalised CDFs for each
 *        of the B, G, R channels.
 *
 * @param bgr  Input BGR image (CV_8UC3).
 * @return     Map with keys "B", "G", "R", each holding a
 *             ChannelHistData containing:
 *               - histogram: 256 integer counts.
 *               - cdf: 256-element normalised CDF in [0, 1].
 */
std::map<std::string, ChannelHistData>
HistogramProcessor::computeChannelHistograms(const cv::Mat& bgr) {
    std::map<std::string, ChannelHistData> result;

    // Split into individual B, G, R planes
    std::vector<cv::Mat> channels = ColorProcessor::splitChannels(bgr);
    const char* names[] = {"B", "G", "R"};

    for (int c = 0; c < 3; ++c) {
        result[names[c]] = computeHistogramAndCDF(channels[c]);
    }
    return result;
}

// ════════════════════════════════════════════════════════════════════
//  Public — computeHistogramAndCDF
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Compute the 256-bin histogram and normalised CDF for a single channel.
 *
 * @param channel Single-channel input (CV_8UC1).
 * @return ChannelHistData containing the histogram and CDF.
 */
ChannelHistData HistogramProcessor::computeHistogramAndCDF(const cv::Mat& channel) {
    ChannelHistData data;
    data.histogram.assign(256, 0);

    // Count pixel intensities into 256 bins
    for (int i = 0; i < channel.rows; ++i) {
        const uchar* row = channel.ptr<uchar>(i);
        for (int j = 0; j < channel.cols; ++j)
            data.histogram[row[j]]++;
    }

    // Build the cumulative distribution function (CDF)
    data.cdf.resize(256);
    double cumSum = 0;
    double totalPixels = channel.rows * channel.cols;
    for (int k = 0; k < 256; ++k) {
        cumSum += data.histogram[k];
        data.cdf[k] = cumSum / totalPixels;   // Normalise to [0, 1]
    }

    return data;
}

// ════════════════════════════════════════════════════════════════════
//  Private — single-channel equalisation
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Equalise a single grayscale channel using CDF-based mapping.
 *
 * Algorithm:
 *   1. Compute the 256-bin histogram and normalised CDF.
 *   3. Find cdf_min_norm (the first non-zero CDF value).
 *   4. Build a 256-entry LUT:
 *        lut[k] = round( (cdf_norm[k] − cdf_min_norm) / (1.0 − cdf_min_norm) × 255 )
 *   5. Map every pixel through the LUT.
 *
 * @param channel  Single-channel input (CV_8UC1).
 * @return         Equalised CV_8UC1 channel.
 */
cv::Mat HistogramProcessor::equalizeChannel(const cv::Mat& channel) {
    // Step 1 & 2: Compute histogram and normalised CDF
    ChannelHistData data = computeHistogramAndCDF(channel);

    // Step 3: Find cdf_min_norm — the first non-zero cumulative count
    double cdfMinNorm = 0;
    for (int k = 0; k < 256; ++k) {
        if (data.cdf[k] > 0) { cdfMinNorm = data.cdf[k]; break; }
    }

    // Denominator for the LUT mapping
    double denom = 1.0 - cdfMinNorm;
    if (denom <= 0) return channel.clone();   // All pixels identical — nothing to do

    // Step 4: Build the equalisation look-up table
    uchar lut[256];
    for (int k = 0; k < 256; ++k)
        lut[k] = cv::saturate_cast<uchar>(
                    std::round((data.cdf[k] - cdfMinNorm) / denom * 255.0));

    // Step 5: Apply the LUT to every pixel
    cv::Mat out(channel.size(), CV_8UC1);
    for (int i = 0; i < channel.rows; ++i) {
        const uchar* src = channel.ptr<uchar>(i);
        uchar* dst = out.ptr<uchar>(i);
        for (int j = 0; j < channel.cols; ++j)
            dst[j] = lut[src[j]];
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  Private — BGR equalisation (luminance-based, matching Python)
//
//  Strategy: equalise the luminance channel, then scale each BGR
//  component proportionally so that colour ratios are preserved.
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Equalise a BGR colour image while preserving colour ratios.
 *
 * Steps:
 *   1. Convert to grayscale luminance L (BT.601 formula).
 *   2. Equalise L → L_eq.
 *   3. For each pixel:  C_out = C × (L_eq / (L + ε))
 *      where ε avoids division by zero on black pixels.
 *
 * @param bgr  Input BGR image (CV_8UC3).
 * @return     Equalised CV_8UC3 image.
 */
cv::Mat HistogramProcessor::equalizeBGR(const cv::Mat& bgr) {
    // Compute grayscale luminance and its equalised version
    cv::Mat L = ColorProcessor::toGrayscale(bgr);
    cv::Mat Leq = equalizeChannel(L);

    // Convert to float for precise division
    cv::Mat Lf, Leqf;
    L.convertTo(Lf, CV_64F);
    Leq.convertTo(Leqf, CV_64F);

    const double eps = 1e-8;   // Small epsilon to prevent division by zero

    // Scale each BGR channel: C_out = C × (L_eq / (L + eps))
    cv::Mat result(bgr.size(), bgr.type());
    for (int i = 0; i < bgr.rows; ++i) {
        const uchar* src  = bgr.ptr<uchar>(i);
        uchar* dst        = result.ptr<uchar>(i);
        const double* lp  = Lf.ptr<double>(i);     // Original luminance
        const double* leq = Leqf.ptr<double>(i);   // Equalised luminance
        for (int j = 0; j < bgr.cols; ++j) {
            double scale = leq[j] / (lp[j] + eps);
            // Apply scale to each of the 3 channels (B, G, R)
            for (int c = 0; c < 3; ++c) {
                double val = src[j * 3 + c] * scale;
                dst[j * 3 + c] = cv::saturate_cast<uchar>(val);
            }
        }
    }
    return result;
}
