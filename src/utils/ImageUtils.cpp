/**
 * @file ImageUtils.cpp
 * @brief Shared low-level image-processing utilities used across
 *        multiple processors.
 *
 * Provides:
 *   - Input validation (empty images, kernel sizes).
 *   - Grayscale conversion helper.
 *   - Border padding (reflect and zero).
 *   - 2-D cross-correlation (matching Python's scipy correlate2d).
 *   - Convenience reflect-pad + correlate + clip pipeline.
 *   - FFT quadrant-swap (fftShift) for frequency-domain processing.
 *
 * Centralising these helpers avoids code duplication and ensures every
 * processor uses the same well-tested primitives.
 */

#include "utils/ImageUtils.h"
#include "processors/ColorProcessor.h"
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════
//  Validation
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Throw std::invalid_argument if the image is empty.
 *
 * This guard is placed at the entry of every public processing function
 * to fail fast with a meaningful message instead of silently producing
 * incorrect results or segfaulting on empty data.
 *
 * @param img      The image to validate.
 * @param context  Optional human-readable context string appended to
 *                 the error message (e.g. the calling function's name).
 *
 * @throws std::invalid_argument  If img.empty() is true.
 */
void ImageUtils::assertNotEmpty(const cv::Mat& img,
                                const std::string& context) {
    if (img.empty()) {
        std::string msg = "Image is empty";
        if (!context.empty()) msg += " (" + context + ")";
        throw std::invalid_argument(msg);
    }
}

/**
 * @brief Validate that a kernel size is odd and at least 3.
 *
 * Most spatial filters require an odd kernel dimension so that the
 * kernel has a well-defined centre pixel.  Sizes smaller than 3 are
 * too small to be useful for smoothing or derivative estimation.
 *
 * @param kernelSize  The proposed kernel side length.
 *
 * @throws std::invalid_argument  If kernelSize < 3 or is even.
 */
void ImageUtils::validateKernelSize(int kernelSize) {
    if (kernelSize < 3 || kernelSize % 2 == 0)
        throw std::invalid_argument(
            "Invalid kernel size (must be odd >= 3), got "
            + std::to_string(kernelSize));
}

// ═══════════════════════════════════════════════════════════════════
//  Colour helpers
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Return a guaranteed single-channel grayscale image.
 *
 * If the input already has one channel it is cloned and returned
 * directly (no conversion).  Otherwise it is converted using
 * ColorProcessor::toGrayscale (BT.601 luminance formula).
 *
 * @param image  Input image (CV_8UC1 or CV_8UC3).
 * @return       A new CV_8UC1 grayscale image.
 */
cv::Mat ImageUtils::ensureGrayscale(const cv::Mat& image) {
    if (image.channels() == 1)
        return image.clone();
    return ColorProcessor::toGrayscale(image);
}

// ═══════════════════════════════════════════════════════════════════
//  Padding
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Pad a single-channel image using reflection at the borders.
 *
 * Reflect padding mirrors pixel values at the image boundary, which
 * avoids introducing artificial edges and is the preferred strategy
 * for most spatial filters (blur, sharpen, etc.).
 *
 * @param channel  Input single-channel Mat (any depth).
 * @param padH     Number of rows to add on the top AND bottom.
 * @param padW     Number of columns to add on the left AND right.
 * @return         A new Mat with dimensions
 *                 (channel.rows + 2*padH) × (channel.cols + 2*padW).
 */
cv::Mat ImageUtils::padReflect(const cv::Mat& channel, int padH, int padW) {
    cv::Mat padded;
    cv::copyMakeBorder(channel, padded,
                       padH, padH, padW, padW,
                       cv::BORDER_REFLECT);
    return padded;
}

/**
 * @brief Pad a single-channel image with zeros (constant black border).
 *
 * Zero padding is appropriate for correlation / convolution operations
 * where the boundary should not contribute extra energy (e.g. gradient
 * computation in edge detection).
 *
 * @param channel  Input single-channel Mat (any depth).
 * @param padH     Number of zero-rows to add on the top AND bottom.
 * @param padW     Number of zero-columns to add on the left AND right.
 * @return         A new Mat with dimensions
 *                 (channel.rows + 2*padH) × (channel.cols + 2*padW).
 */
cv::Mat ImageUtils::padZero(const cv::Mat& channel, int padH, int padW) {
    cv::Mat padded;
    cv::copyMakeBorder(channel, padded,
                       padH, padH, padW, padW,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
    return padded;
}

// ═══════════════════════════════════════════════════════════════════
//  2-D cross-correlation
//  (matches Python's EdgeDetectorProcessor._convolve2d)
//
//  Input channel must be CV_64F.  The kernel can be any size
//  (even or odd).  Uses zero-padding; the result has the same
//  dimensions as the input channel ("same" mode).
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Perform 2-D cross-correlation of a channel with a kernel.
 *
 * Cross-correlation slides the kernel over the image *without* flipping
 * it (unlike convolution, which flips both axes).  This matches the
 * behaviour of Python's scipy.signal.correlate2d(mode='same').
 *
 * The input is zero-padded so that the output has the same spatial
 * dimensions as the input ("same" mode).
 *
 * @param channel  Single-channel input image (must be CV_64F).
 * @param kernel   Correlation kernel (must be CV_64F, any size).
 * @return         CV_64F result, same rows/cols as channel.
 *
 * @note Complexity is O(H × W × kh × kw) — a brute-force implementation
 *       kept for clarity and correctness rather than speed.
 */
cv::Mat ImageUtils::correlate2d(const cv::Mat& channel,
                                const cv::Mat& kernel) {
    // Kernel dimensions
    int kh = kernel.rows, kw = kernel.cols;

    // Padding needed to keep the output the same size as the input
    int padH = kh / 2, padW = kw / 2;

    // Original image dimensions
    int H = channel.rows, W = channel.cols;

    // Zero-pad the input so the kernel can be centred on every pixel
    cv::Mat padded = padZero(channel, padH, padW);

    // Allocate the output matrix (same size as channel)
    cv::Mat result(H, W, CV_64F);

    // Slide the kernel over every position in the original image
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double sum = 0;
            // Element-wise multiply-accumulate over the kernel window
            for (int m = 0; m < kh; ++m)
                for (int n = 0; n < kw; ++n)
                    sum += padded.at<double>(i + m, j + n)
                         * kernel.at<double>(m, n);
            result.at<double>(i, j) = sum;
        }
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════
//  Convenience: pad (reflect) → correlate → clip to uint8
//  Expects a CV_8UC1 channel and a CV_64F kernel.
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Apply a floating-point kernel to a uint8 channel using
 *        reflect padding, returning a uint8 result.
 *
 * This is a convenience pipeline that:
 *   1. Reflect-pads the input channel.
 *   2. Slides the kernel over every pixel, computing the weighted sum
 *      directly on uchar data (cast to double for arithmetic).
 *   3. Clips the result to [0, 255] via cv::saturate_cast.
 *
 * Useful for filters like Gaussian blur or box blur where reflection
 * padding and 8-bit output are the expected behaviour.
 *
 * @param channel  Single-channel input (CV_8UC1).
 * @param kernel   Convolution/correlation kernel (CV_64F, any size).
 * @return         CV_8UC1 result, same size as channel.
 */
cv::Mat ImageUtils::applyKernelReflect(const cv::Mat& channel,
                                       const cv::Mat& kernel) {
    // Kernel dimensions
    int kh = kernel.rows, kw = kernel.cols;

    // Compute reflect-padding amounts (half the kernel on each side)
    int padH = kh / 2, padW = kw / 2;

    // Pad the input using reflection at the borders
    cv::Mat padded = padReflect(channel, padH, padW);

    // Allocate output (same size as original, 8-bit single channel)
    cv::Mat output(channel.rows, channel.cols, CV_8UC1);

    // Slide the kernel and compute the weighted sum at each pixel
    for (int i = 0; i < channel.rows; ++i) {
        for (int j = 0; j < channel.cols; ++j) {
            double val = 0;
            for (int m = 0; m < kh; ++m)
                for (int n = 0; n < kw; ++n)
                    val += padded.at<uchar>(i + m, j + n)
                         * kernel.at<double>(m, n);
            // Clip to [0, 255] and store as uchar
            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(val);
        }
    }
    return output;
}

// ═══════════════════════════════════════════════════════════════════
//  fftShift — swap diagonally opposite quadrants (in-place)
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Swap diagonally opposite quadrants of a matrix in-place.
 *
 * After a 2-D DFT the DC component (zero frequency) sits at the
 * top-left corner.  fftShift rearranges the quadrants so that DC
 * moves to the centre, which is more natural for circular mask
 * operations and visualisation.
 *
 * Quadrant layout before shift:
 *   ┌───┬───┐
 *   │ Q0│ Q1│     Q0 = top-left,  Q1 = top-right
 *   ├───┼───┤     Q2 = bottom-left, Q3 = bottom-right
 *   │ Q2│ Q3│
 *   └───┴───┘
 *
 * After shift:  Q0 ↔ Q3  and  Q1 ↔ Q2  (diagonal swap).
 *
 * @param mat  Matrix to rearrange in-place.  May be multi-channel
 *             (e.g. 2-channel complex).
 *
 * @note For even-dimensioned matrices, applying fftShift twice
 *       restores the original layout (i.e. ifftShift ≡ fftShift).
 */
void ImageUtils::fftShift(cv::Mat& mat) {
    // Centre coordinates (integer division truncates for odd sizes)
    int cx = mat.cols / 2;
    int cy = mat.rows / 2;

    // Create ROI sub-matrices referencing the four quadrants
    cv::Mat q0(mat, cv::Rect(0,  0,  cx, cy));   // top-left
    cv::Mat q1(mat, cv::Rect(cx, 0,  cx, cy));   // top-right
    cv::Mat q2(mat, cv::Rect(0,  cy, cx, cy));   // bottom-left
    cv::Mat q3(mat, cv::Rect(cx, cy, cx, cy));   // bottom-right

    // Swap Q0 ↔ Q3  (top-left ↔ bottom-right)
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);

    // Swap Q1 ↔ Q2  (top-right ↔ bottom-left)
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}
