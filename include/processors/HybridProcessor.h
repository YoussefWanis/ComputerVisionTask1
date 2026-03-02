#ifndef HYBRIDPROCESSOR_H
#define HYBRIDPROCESSOR_H

#include <opencv2/opencv.hpp>

/**
 * HybridProcessor — combine two images in the frequency domain.
 * Mirrors Python's Processors/hybird.py  create_hybrid_image().
 *
 * High-pass = hpImage − lowpass(hpImage), keeping it zero-centred.
 */
class HybridProcessor {
public:
    /**
     * @param lpImage    Low-pass source  (BGR, same size as hpImage).
     * @param hpImage    High-pass source (BGR, same size as lpImage).
     * @param cutoffLow  Radius for the LP filter on lpImage.
     * @param cutoffHigh Radius for the LP filter used to derive HP of hpImage.
     */
    static cv::Mat create(const cv::Mat& lpImage,
                          const cv::Mat& hpImage,
                          int cutoffLow  = 30,
                          int cutoffHigh = 30);

private:
    HybridProcessor() = delete;
};

#endif // HYBRIDPROCESSOR_H
