/**
 * @file HybridProcessor.cpp
 * @brief Creates hybrid images by combining frequency-domain content
 *        from two source images.
 *
 * A hybrid image merges the low-frequency content of one image with
 * the high-frequency content of another.  The result appears different
 * depending on viewing distance:
 *   - Close up  → the high-frequency details dominate perception.
 *   - Far away  → only the low-frequency structure is visible.
 *
 * Mirrors the Python function create_hybrid_image() from
 * Processors/hybrid.py.
 */

#include "processors/HybridProcessor.h"
#include "processors/FFTProcessor.h"
#include "utils/ImageUtils.h"
#include <algorithm>

// ════════════════════════════════════════════════════════════════════
//  create — matches Python's create_hybrid_image()
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Build a hybrid image from two source images.
 *
 * The formula is:
 *     hybrid = lowpass(lpImage, cutoffLow)
 *            + ( hpImage − lowpass(hpImage, cutoffHigh) )
 *
 * where the second term is a high-pass residual (zero-centred), so
 * values can be negative before the final clipping stage.
 *
 * Processing is performed per-channel (supports both BGR and grayscale)
 * via FFTProcessor::processRaw(), which returns unnormalised CV_64F
 * results suitable for arithmetic combination.
 *
 * @param lpImage     Low-pass source image (BGR or grayscale).
 *                    Must be the same size as hpImage.
 * @param hpImage     High-pass source image (BGR or grayscale).
 *                    Must be the same size as lpImage.
 * @param cutoffLow   Frequency-domain radius (in pixels) for the ideal
 *                    low-pass filter applied to lpImage.  Larger values
 *                    preserve more low-frequency detail.
 * @param cutoffHigh  Frequency-domain radius (in pixels) for the ideal
 *                    low-pass filter used to derive the high-pass
 *                    component of hpImage.  Smaller values retain more
 *                    high-frequency detail.
 * @return            CV_8UC3 (or CV_8UC1) hybrid image, pixel values
 *                    clipped to [0, 255].
 *
 * @throws std::invalid_argument  If either input image is empty.
 */
cv::Mat HybridProcessor::create(const cv::Mat& lpImage,
                                const cv::Mat& hpImage,
                                int cutoffLow,
                                int cutoffHigh) {
    // Validate that neither source image is empty
    ImageUtils::assertNotEmpty(lpImage, "HybridProcessor LP image");
    ImageUtils::assertNotEmpty(hpImage, "HybridProcessor HP image");

    FFTProcessor fft;

    // ── Low-pass component ─────────────────────────────────────
    // Apply an ideal circular low-pass filter to lpImage in the
    // frequency domain; result is CV_64F (unnormalised).
    cv::Mat lowA = fft.processRaw(lpImage, "lowpass", cutoffLow);

    // ── High-pass component ────────────────────────────────────
    // High-pass = original − lowpass(original).
    // First, get the low-pass of hpImage:
    cv::Mat lowB = fft.processRaw(hpImage, "lowpass", cutoffHigh);

    // Convert hpImage to CV_64F so we can subtract the low-pass result
    cv::Mat hpB;
    hpImage.convertTo(hpB, CV_64F);

    // Subtract to obtain zero-centred high-frequency residual
    hpB = hpB - lowB;

    // ── Combine ────────────────────────────────────────────────
    // Add the two frequency components together
    cv::Mat hybrid = lowA + hpB;

    // ── Clip and convert ───────────────────────────────────────
    // saturate_cast inside convertTo clips negative values to 0
    // and values > 255 to 255, then rounds to uint8
    cv::Mat out;
    hybrid.convertTo(out, CV_8U);
    return out;
}
