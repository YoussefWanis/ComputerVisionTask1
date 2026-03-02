#ifndef FFTPROCESSOR_H
#define FFTPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include "utils/ImageUtils.h"

/**
 * FFTProcessor — ideal circular low-pass / high-pass in the frequency domain.
 * Mirrors Python's Processors/fft.py.
 */
class FFTProcessor {
public:
    FFTProcessor() = default;

    /** Filter + normalise to uint8 [0, 255] for display. */
    cv::Mat process(const cv::Mat& image,
                    const std::string& filterType,
                    int cutoffRadius);

    /** Filter → raw float64 (for hybrid-image building). */
    cv::Mat processRaw(const cv::Mat& image,
                       const std::string& filterType,
                       int cutoffRadius);

private:
    cv::Mat makeCircularMask(cv::Size shape,
                             const std::string& filterType,
                             int cutoffRadius);

    cv::Mat filterChannelRaw(const cv::Mat& channel,
                             const std::string& filterType,
                             int cutoffRadius);

    cv::Mat filterChannel(const cv::Mat& channel,
                          const std::string& filterType,
                          int cutoffRadius);
};

#endif // FFTPROCESSOR_H
