/**
 * @file HistogramProcessor.h
 * @brief Declaration of the HistogramProcessor class and the
 *        ChannelHistData helper struct.
 *
 * Mirrors the Python implementation in Processors/histogram.py.
 * Provides histogram equalisation (grayscale & luminance-based BGR),
 * min-max normalisation, and per-channel histogram/CDF computation.
 */

#ifndef HISTOGRAMPROCESSOR_H
#define HISTOGRAMPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>

/**
 * @struct ChannelHistData
 * @brief Holds the histogram and cumulative distribution function (CDF)
 *        for a single colour channel.
 */
struct ChannelHistData {
    std::vector<int>    histogram;   ///< 256-bin intensity histogram (counts per level).
    std::vector<double> cdf;         ///< Normalised CDF in [0, 1], 256 elements.
};

/**
 * @class HistogramProcessor
 * @brief Histogram-based image analysis and enhancement.
 *
 * Key operations:
 *   - **Equalise** — stretches the histogram to span the full [0, 255]
 *     range.  For colour images the luminance channel is equalised and
 *     the result is mapped back to BGR to preserve hue.
 *   - **Normalize** — applies simple min-max scaling to [0, 255].
 *   - **Channel histograms** — computes 256-bin histograms and CDFs
 *     for each of the B, G, R channels of a colour image.
 */
class HistogramProcessor {
public:
    /** @brief Default constructor. */
    HistogramProcessor() = default;

    /**
     * @brief Equalise the histogram of a grayscale or BGR image.
     *
     * For grayscale inputs the single channel is equalised directly.
     * For BGR inputs, luminance is equalised and the per-pixel colour
     * ratios are preserved (luminance-based equalisation).
     *
     * @param image  Input image (CV_8UC1 or CV_8UC3).
     * @return       Equalised image of the same type and size.
     *
     * @throws std::invalid_argument  If the image is empty.
     */
    cv::Mat equalize(const cv::Mat& image);

    /**
     * @brief Min-max normalise all channels to span [0, 255].
     *
     * Finds the global minimum and maximum across all channels and
     * linearly maps the range [min, max] → [0, 255].
     *
     * @param image  Input image (any type supported by cv::minMaxLoc).
     * @return       Normalised CV_8U image.
     */
    static cv::Mat normalize(const cv::Mat& image);

    /**
     * @brief Compute the 256-bin histogram and normalised CDF for a single channel.
     * @param channel Single-channel input (CV_8UC1).
     * @return ChannelHistData containing the histogram and CDF.
     */
    static ChannelHistData computeHistogramAndCDF(const cv::Mat& channel);

    /**
     * @brief Compute per-channel histograms and CDFs for a BGR image.
     *
     * @param bgr  Input BGR image (CV_8UC3).
     * @return     Map with keys "B", "G", "R", each mapping to a
     *             ChannelHistData containing a 256-bin histogram and
     *             a normalised CDF.
     */
    std::map<std::string, ChannelHistData>
    computeChannelHistograms(const cv::Mat& bgr);

private:
    /**
     * @brief Equalise a single grayscale channel using CDF mapping.
     * @param channel  Single-channel input (CV_8UC1).
     * @return         Equalised CV_8UC1 channel.
     */
    static cv::Mat equalizeChannel(const cv::Mat& channel);

    /**
     * @brief Equalise a BGR image by operating on its luminance.
     *
     * Computes grayscale luminance, equalises it, then scales each
     * BGR channel proportionally so colour ratios are preserved.
     *
     * @param bgr  Input BGR image (CV_8UC3).
     * @return     Equalised CV_8UC3 image.
     */
    cv::Mat equalizeBGR(const cv::Mat& bgr);
};

#endif // HISTOGRAMPROCESSOR_H
