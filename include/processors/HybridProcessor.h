/**
 * @file HybridProcessor.h
 * @brief Declaration of the HybridProcessor static class for creating
 *        hybrid images by combining frequency-domain content from
 *        two source images.
 *
 * Mirrors the Python function create_hybrid_image() from
 * Processors/hybrid.py.
 *
 * A hybrid image merges the low-frequency structure of one image
 * with the high-frequency detail of another.  At close range the
 * high-frequency content dominates perception; from far away only
 * the low-frequency content is visible.
 */

#ifndef HYBRIDPROCESSOR_H
#define HYBRIDPROCESSOR_H

#include <opencv2/opencv.hpp>

/**
 * @class HybridProcessor
 * @brief Creates hybrid images by combining frequency-domain content
 *        of two source images.
 *
 * The formula is:
 *   hybrid = lowpass(lpImage, cutoffLow)
 *          + ( hpImage − lowpass(hpImage, cutoffHigh) )
 *
 * Cannot be instantiated — provides a single static method.
 */
class HybridProcessor {
public:
    /**
     * @brief Build a hybrid image from two source images.
     *
     * Both images must have the same spatial dimensions.  The LP
     * and HP sources can be either BGR or grayscale.
     *
     * @param lpImage    Low-pass source image (supplies the blurry
     *                   / large-scale structure).
     * @param hpImage    High-pass source image (supplies the sharp
     *                   / fine-detail component).
     * @param cutoffLow  Frequency radius for the LP filter on lpImage.
     *                   Larger → more low-frequency content preserved.
     * @param cutoffHigh Frequency radius for the LP filter used to
     *                   derive the HP component of hpImage.
     *                   Smaller → more high-frequency content retained.
     * @return           CV_8U hybrid image, clipped to [0, 255].
     *
     * @throws std::invalid_argument  If either input image is empty.
     */
    static cv::Mat create(const cv::Mat& lpImage,
                          const cv::Mat& hpImage,
                          int cutoffLow  = 30,
                          int cutoffHigh = 30);

private:
    HybridProcessor() = delete;   ///< Pure-static class — cannot be instantiated.
};

#endif // HYBRIDPROCESSOR_H
