/**
 * @file ImageUtils.h
 * @brief Declaration of the ImageUtils static utility class.
 *
 * Provides shared low-level image-processing helpers used across
 * multiple processor classes: input validation, grayscale conversion,
 * border padding, 2-D cross-correlation, and FFT quadrant swapping.
 *
 * This is the C++ equivalent of the common patterns that appear in the
 * Python code-base (padding, convolution, grayscale conversion, etc.).
 * Centralising them avoids duplication and keeps every processor consistent.
 */

#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>

/**
 * @class ImageUtils
 * @brief Pure-static collection of shared low-level image helpers.
 *
 * Cannot be instantiated — all methods are static.  Grouped into:
 *   - **Validation** — assertNotEmpty, validateKernelSize.
 *   - **Colour**     — ensureGrayscale.
 *   - **Padding**    — padReflect, padZero.
 *   - **Correlation** — correlate2d, applyKernelReflect.
 *   - **FFT**        — fftShift.
 */
class ImageUtils {
public:
    // ── Image validation ────────────────────────────────────

    /**
     * @brief Throw std::invalid_argument if the image is empty.
     * @param img      Image to check.
     * @param context  Optional label for the error message (e.g. caller name).
     * @throws std::invalid_argument  If img.empty() is true.
     */
    static void assertNotEmpty(const cv::Mat& img,
                               const std::string& context = "");

    /**
     * @brief Throw if kernelSize is even or less than 3.
     * @param kernelSize  Proposed kernel side length.
     * @throws std::invalid_argument  If the size is invalid.
     */
    static void validateKernelSize(int kernelSize);

    // ── Colour helpers ──────────────────────────────────────

    /**
     * @brief Return a single-channel grayscale Mat.
     *
     * If the image is already 1-channel it is cloned unchanged.
     * Otherwise it is converted via ColorProcessor::toGrayscale
     * (BT.601 luminance formula).
     *
     * @param image  Input image (CV_8UC1 or CV_8UC3).
     * @return       A new CV_8UC1 grayscale image.
     */
    static cv::Mat ensureGrayscale(const cv::Mat& image);

    // ── Padding ─────────────────────────────────────────────

    /**
     * @brief Reflect-pad a single-channel Mat by (padH, padW) on each side.
     * @param channel  Input single-channel Mat.
     * @param padH     Rows to add top and bottom.
     * @param padW     Columns to add left and right.
     * @return         Padded Mat of size (rows+2*padH, cols+2*padW).
     */
    static cv::Mat padReflect(const cv::Mat& channel, int padH, int padW);

    /**
     * @brief Zero-pad (BORDER_CONSTANT=0) a single-channel Mat.
     * @param channel  Input single-channel Mat.
     * @param padH     Rows to add top and bottom.
     * @param padW     Columns to add left and right.
     * @return         Padded Mat of size (rows+2*padH, cols+2*padW).
     */
    static cv::Mat padZero(const cv::Mat& channel, int padH, int padW);

    // ── Convolution / Correlation ───────────────────────────

    /**
     * @brief 2-D cross-correlation of a single-channel image with a kernel.
     *
     * No kernel flip (correlation, not convolution).  Uses zero-padding;
     * result has the same dimensions as channel ("same" mode).
     * Matches the Python _convolve2d helper.
     *
     * @param channel  CV_64F single-channel input.
     * @param kernel   CV_64F kernel (any size, even or odd).
     * @return         CV_64F result, same size as channel.
     */
    static cv::Mat correlate2d(const cv::Mat& channel,
                               const cv::Mat& kernel);

    /**
     * @brief Apply a CV_64F kernel to a uint8 channel with reflect
     *        padding and return a CV_8U result clipped to [0, 255].
     *
     * Convenience wrapper: padReflect → weighted sum → saturate_cast.
     *
     * @param channel  Single-channel input (CV_8UC1).
     * @param kernel   Convolution/correlation kernel (CV_64F).
     * @return         CV_8UC1 result, same size as channel.
     */
    static cv::Mat applyKernelReflect(const cv::Mat& channel,
                                      const cv::Mat& kernel);

    // ── FFT helpers ─────────────────────────────────────────

    /**
     * @brief Swap diagonally-opposite quadrants of a matrix in-place.
     *
     * Moves the DC component from the top-left corner to the centre
     * (or vice versa).  Works on multi-channel matrices (e.g. 2-ch
     * complex).  For even dimensions, ifftShift ≡ fftShift.
     *
     * @param mat  Matrix to rearrange in-place.
     */
    static void fftShift(cv::Mat& mat);

private:
    ImageUtils() = delete;   ///< Pure-static class — cannot be instantiated.
};

#endif // IMAGEUTILS_H
