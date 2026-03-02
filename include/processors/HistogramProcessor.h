#ifndef HISTOGRAMPROCESSOR_H
#define HISTOGRAMPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <map>
#include <string>
#include <vector>

/** Per-channel histogram data (256 bins). */
struct ChannelHistData {
    std::vector<int>    histogram;   ///< 256-element counts.
    std::vector<double> cdf;         ///< Normalised CDF in [0, 1].
};

/**
 * HistogramProcessor — histogram equalisation & analysis.
 * Mirrors Python's Processors/histogram.py.
 */
class HistogramProcessor {
public:
    HistogramProcessor() = default;

    /** Equalise (grayscale or luminance-based RGB). */
    cv::Mat equalize(const cv::Mat& image);

    /** Min-max normalise to [0, 255]. */
    static cv::Mat normalize(const cv::Mat& image);

    /**
     * Per-channel histograms + CDFs.
     * Keys: "B", "G", "R".
     */
    std::map<std::string, ChannelHistData>
    computeChannelHistograms(const cv::Mat& bgr);

private:
    cv::Mat equalizeChannel(const cv::Mat& channel);
    cv::Mat equalizeBGR    (const cv::Mat& bgr);
};

#endif // HISTOGRAMPROCESSOR_H
