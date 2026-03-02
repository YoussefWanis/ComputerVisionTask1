/**
 * @file FFTProcessor.cpp
 * @brief Frequency-domain image filtering using ideal circular
 *        low-pass and high-pass masks.
 *
 * Mirrors the Python implementation in Processors/fft.py.
 *
 * The pipeline for each channel is:
 *   1. Forward DFT (fft2).
 *   2. fftShift — move DC to the centre.
 *   3. Multiply the spectrum by a binary circular mask.
 *   4. ifftShift — undo the shift.
 *   5. Inverse DFT (ifft2).
 *   6. Take the magnitude (abs) of the complex result.
 *
 * Two public variants are provided:
 *   - process()    → normalised CV_8U output for display.
 *   - processRaw() → unnormalised CV_64F output for arithmetic
 *                     combination (used by HybridProcessor).
 *
 * Multi-channel (BGR) images are handled by splitting, filtering
 * each channel independently, and merging back.
 */

#include "processors/FFTProcessor.h"
#include "utils/ImageUtils.h"
#include <cmath>
#include <stdexcept>

// ════════════════════════════════════════════════════════════════════
//  Public — process (normalised uint8 for display)
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Apply an ideal circular frequency filter and return a
 *        normalised 8-bit image suitable for display.
 *
 * For 3-channel (BGR) images, each channel is filtered independently
 * and the results are merged back into a single colour image.
 *
 * @param image        Input image (CV_8UC1 or CV_8UC3). Must not be empty.
 * @param filterType   "lowpass" or "highpass".
 * @param cutoffRadius Radius (in pixels) of the circular mask centred
 *                     on the DC component.
 * @return             CV_8U image with pixel values mapped to [0, 255].
 *
 * @throws std::invalid_argument  If the image is empty.
 */
cv::Mat FFTProcessor::process(const cv::Mat& image,
                              const std::string& filterType,
                              int cutoffRadius) {
    ImageUtils::assertNotEmpty(image, "FFTProcessor::process");

    // For multi-channel images: split → filter each → merge
    if (image.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        for (auto& ch : channels)
            ch = filterChannel(ch, filterType, cutoffRadius);
        cv::Mat result;
        cv::merge(channels, result);
        return result;
    }

    // Single-channel path
    return filterChannel(image, filterType, cutoffRadius);
}

// ════════════════════════════════════════════════════════════════════
//  Public — processRaw (float64, for hybrid images)
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Apply an ideal circular frequency filter and return the raw
 *        (unnormalised) CV_64F result.
 *
 * The output preserves the true pixel magnitude values, which is
 * essential when combining filtered outputs arithmetically (e.g. in
 * HybridProcessor::create where low-pass and high-pass results are
 * added together).
 *
 * @param image        Input image (CV_8UC1 or CV_8UC3). Must not be empty.
 * @param filterType   "lowpass" or "highpass".
 * @param cutoffRadius Radius (in pixels) of the circular mask.
 * @return             CV_64F image (single- or multi-channel).
 *
 * @throws std::invalid_argument  If the image is empty.
 */
cv::Mat FFTProcessor::processRaw(const cv::Mat& image,
                                 const std::string& filterType,
                                 int cutoffRadius) {
    ImageUtils::assertNotEmpty(image, "FFTProcessor::processRaw");

    // For multi-channel images: split → filter each → merge
    if (image.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        for (auto& ch : channels)
            ch = filterChannelRaw(ch, filterType, cutoffRadius);
        cv::Mat result;
        cv::merge(channels, result);
        return result;
    }

    // Single-channel path
    return filterChannelRaw(image, filterType, cutoffRadius);
}

// ════════════════════════════════════════════════════════════════════
//  Circular mask  (matching Python's _make_circular_mask)
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Generate a binary circular frequency-domain mask.
 *
 * Creates a CV_64F matrix of the given shape where each element is
 * either 1.0 (pass) or 0.0 (reject) based on the Euclidean distance
 * from the pixel to the centre of the image.
 *
 * For "lowpass":  pass = 1 if distance ≤ cutoffRadius, else 0.
 * For "highpass": pass = 1 if distance >  cutoffRadius, else 0.
 *
 * The mask is designed to be applied element-wise to an fftShifted
 * spectrum (i.e. DC at the centre).
 *
 * @param shape         Size of the mask (rows × cols), matching the
 *                      spatial dimensions of the image being filtered.
 * @param filterType    "lowpass" or "highpass".
 * @param cutoffRadius  Radius of the circle in pixels.
 * @return              CV_64F mask with values 0.0 or 1.0.
 */
cv::Mat FFTProcessor::makeCircularMask(cv::Size shape,
                                       const std::string& filterType,
                                       int cutoffRadius) {
    int H = shape.height, W = shape.width;

    // Centre of the frequency plane (DC location after fftShift)
    int cy = H / 2, cx = W / 2;

    cv::Mat mask(H, W, CV_64F);

    for (int i = 0; i < H; ++i) {
        double* row = mask.ptr<double>(i);
        for (int j = 0; j < W; ++j) {
            // Euclidean distance from pixel (j, i) to the centre (cx, cy)
            double dist = std::sqrt(static_cast<double>((j - cx) * (j - cx)
                                  + (i - cy) * (i - cy)));
            if (filterType == "lowpass")
                row[j] = (dist <= cutoffRadius) ? 1.0 : 0.0;
            else  // "highpass"
                row[j] = (dist > cutoffRadius) ? 1.0 : 0.0;
        }
    }
    return mask;
}

// ════════════════════════════════════════════════════════════════════
//  Core FFT pipeline — raw float64 output
//
//  Steps (matching the Python implementation exactly):
//    1. fft2        — forward discrete Fourier transform
//    2. fftshift    — move DC to centre
//    3. mask ×      — element-wise multiply with circular mask
//    4. ifftshift   — reverse the shift (same as fftshift for even dims)
//    5. ifft2       — inverse discrete Fourier transform
//    6. abs         — take magnitude of the complex result
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Filter a single grayscale channel in the frequency domain.
 *
 * Performs the full forward-DFT → mask → inverse-DFT pipeline on one
 * single-channel image and returns the raw CV_64F magnitude result.
 *
 * @param channel       Single-channel input (CV_8UC1).
 * @param filterType    "lowpass" or "highpass".
 * @param cutoffRadius  Radius (pixels) for the circular frequency mask.
 * @return              CV_64F single-channel filtered image (unnormalised).
 */
cv::Mat FFTProcessor::filterChannelRaw(const cv::Mat& channel,
                                       const std::string& filterType,
                                       int cutoffRadius) {
    // Convert input to 64-bit float for DFT precision
    cv::Mat floatCh;
    channel.convertTo(floatCh, CV_64F);

    // Build a 2-channel complex Mat: [real = pixel values, imag = 0]
    cv::Mat planes[] = {floatCh,
                        cv::Mat::zeros(floatCh.size(), CV_64F)};
    cv::Mat complex;
    cv::merge(planes, 2, complex);

    // Step 1: Forward Discrete Fourier Transform
    cv::dft(complex, complex);

    // Step 2: Shift DC component from top-left corner to the centre
    ImageUtils::fftShift(complex);

    // Step 3: Apply the circular frequency mask to both real & imaginary parts
    cv::Mat mask = makeCircularMask(channel.size(), filterType, cutoffRadius);
    cv::split(complex, planes);                       // separate real / imag
    cv::multiply(planes[0], mask, planes[0]);         // mask the real part
    cv::multiply(planes[1], mask, planes[1]);         // mask the imaginary part
    cv::merge(planes, 2, complex);                    // recombine

    // Step 4: Inverse fftShift (identical to fftShift for even dimensions)
    ImageUtils::fftShift(complex);

    // Step 5: Inverse DFT with scaling (DFT_SCALE divides by N×M)
    cv::dft(complex, complex, cv::DFT_INVERSE | cv::DFT_SCALE);

    // Step 6: Compute magnitude = sqrt(real² + imag²) of complex result
    cv::split(complex, planes);
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);

    return magnitude;   // CV_64F, unnormalised
}

// ════════════════════════════════════════════════════════════════════
//  Normalised filter — [0,255] uint8 for display
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Filter a single channel and normalise the result to [0, 255].
 *
 * Calls filterChannelRaw() to obtain the raw floating-point output,
 * then applies min-max normalisation to stretch the value range to
 * [0, 255] for 8-bit display.
 *
 * @param channel       Single-channel input (CV_8UC1).
 * @param filterType    "lowpass" or "highpass".
 * @param cutoffRadius  Radius (pixels) for the circular frequency mask.
 * @return              CV_8UC1 normalised filtered image.
 */
cv::Mat FFTProcessor::filterChannel(const cv::Mat& channel,
                                    const std::string& filterType,
                                    int cutoffRadius) {
    // Get the raw (unnormalised) filtered result
    cv::Mat raw = filterChannelRaw(channel, filterType, cutoffRadius);

    // Find the minimum and maximum pixel values
    double mn, mx;
    cv::minMaxLoc(raw, &mn, &mx);

    // Min-max normalise to [0, 255] (avoid division by zero)
    if (mx > mn)
        raw = (raw - mn) / (mx - mn) * 255.0;

    // Convert from CV_64F to CV_8U (saturate_cast clips any rounding errors)
    cv::Mat out;
    raw.convertTo(out, CV_8U);
    return out;
}

