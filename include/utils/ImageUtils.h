#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>

/**
 * ImageUtils — shared low-level helpers used across multiple processors.
 *
 * This is the C++ equivalent of the common patterns that appear in the
 * Python code-base (padding, convolution, grayscale conversion, etc.).
 * Centralising them avoids duplication and keeps every processor consistent.
 */
class ImageUtils {
public:
    // ── Image validation ────────────────────────────────────────

    /** Throw if the image is empty. */
    static void assertNotEmpty(const cv::Mat& img,
                               const std::string& context = "");

    /** Throw if kernelSize is < 3 or even. */
    static void validateKernelSize(int kernelSize);

    // ── Colour helpers ──────────────────────────────────────────

    /**
     * Return a single-channel grayscale Mat.
     * If the image is already 1-channel it is cloned unchanged.
     * Delegates to ColorProcessor::toGrayscale for 3-channel input.
     */
    static cv::Mat ensureGrayscale(const cv::Mat& image);

    // ── Padding ─────────────────────────────────────────────────

    /** Reflect-pad a single-channel Mat by (padH, padW) on each side. */
    static cv::Mat padReflect(const cv::Mat& channel, int padH, int padW);

    /** Zero-pad (BORDER_CONSTANT) a single-channel Mat. */
    static cv::Mat padZero(const cv::Mat& channel, int padH, int padW);

    // ── Convolution / Correlation ───────────────────────────────

    /**
     * 2-D cross-correlation of a single-channel image with a kernel.
     * (No kernel flip, i.e. correlation — matching the Python _convolve2d.)
     *
     * @param channel  CV_64F single-channel input.
     * @param kernel   CV_64F kernel (any size, even or odd).
     * @return CV_64F result, same size as channel.
     */
    static cv::Mat correlate2d(const cv::Mat& channel,
                               const cv::Mat& kernel);

    /**
     * Apply a CV_64F kernel to a uint8 single-channel image with reflect
     * padding, and return a CV_8U result clipped to [0, 255].
     *
     * Convenience wrapper:  padReflect → correlate2d → saturate_cast.
     */
    static cv::Mat applyKernelReflect(const cv::Mat& channel,
                                      const cv::Mat& kernel);

    // ── FFT helpers ─────────────────────────────────────────────

    /** Swap diagonally-opposite quadrants (in-place). */
    static void fftShift(cv::Mat& mat);

private:
    ImageUtils() = delete;   // pure-static class
};

#endif // IMAGEUTILS_H
