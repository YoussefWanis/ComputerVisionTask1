/**
 * @file FilterProcessor.cpp
 * @brief Implements spatial-domain image filters: average (box),
 *        Gaussian, and median.
 *
 * Mirrors the Python implementation in Processors/filter.py.
 * Each filter operates per-channel on a BGR image: the channels are
 * split, filtered independently, and merged back.  All filters use
 * reflect-padding at the borders to avoid artificial edge artefacts.
 */

#include "processors/FilterProcessor.h"
#include "processors/ColorProcessor.h"
#include "utils/ImageUtils.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

// ════════════════════════════════════════════════════════════════════
//  Public
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Apply the specified spatial filter to a BGR image.
 *
 * Validates inputs, selects the per-channel filter function via a
 * member-function pointer, splits the image into B/G/R planes,
 * filters each plane independently, and merges the results.
 *
 * @param image       Input BGR image (CV_8UC3). Must not be empty.
 * @param filterType  Filter algorithm: "average" | "gaussian" | "median".
 * @param kernelSize  Odd integer ≥ 3 specifying the square kernel width.
 * @return            Filtered CV_8UC3 image of the same size.
 *
 * @throws std::invalid_argument  If the image is empty, the kernel
 *         size is invalid, or the filter type is unrecognised.
 */
cv::Mat FilterProcessor::process(const cv::Mat& image,
                                 const std::string& filterType,
                                 int kernelSize) {
    // Guard against empty images and bad kernel sizes
    ImageUtils::assertNotEmpty(image, "FilterProcessor::process");
    ImageUtils::validateKernelSize(kernelSize);

    // Select the per-channel filter function via member-function pointer
    using Fn = cv::Mat (FilterProcessor::*)(const cv::Mat&, int);
    Fn fn = nullptr;
    if      (filterType == "average")  fn = &FilterProcessor::averageFilter;
    else if (filterType == "gaussian") fn = &FilterProcessor::gaussianFilter;
    else if (filterType == "median")   fn = &FilterProcessor::medianFilter;
    else throw std::invalid_argument("Unknown filter_type: " + filterType);

    // Split into individual colour channels (B, G, R)
    std::vector<cv::Mat> channels = ColorProcessor::splitChannels(image);

    // Apply the chosen filter to each channel independently
    for (auto& ch : channels)
        ch = (this->*fn)(ch, kernelSize);

    // Merge the filtered channels back into a single BGR image
    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

// ════════════════════════════════════════════════════════════════════
//  Gaussian Filter
//  σ = kernelSize / 6  (matching the Python implementation)
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Apply Gaussian-weighted smoothing to a single channel.
 *
 * Steps:
 *   1. Compute σ = k / 6 (so that 3σ ≈ half the kernel width).
 *   2. Build a 1-D Gaussian vector and normalise it to sum = 1.
 *   3. Form the 2-D kernel via outer product of the 1-D vector.
 *   4. Delegate to ImageUtils::applyKernelReflect for reflect-padded
 *      spatial correlation and 8-bit clipping.
 *
 * @param channel  Single-channel input (CV_8UC1).
 * @param k        Odd kernel side length (≥ 3).
 * @return         Smoothed CV_8UC1 channel.
 */
cv::Mat FilterProcessor::gaussianFilter(const cv::Mat& channel, int k) {
    double sigma = k / 6.0;   // Standard deviation of the Gaussian
    int half = k / 2;         // Half-width (kernel centre index)

    // Build 1-D Gaussian kernel and accumulate the normalisation sum
    std::vector<double> g1d(k);
    double sum = 0;
    for (int i = 0; i < k; ++i) {
        double x = i - half;                                     // Distance from centre
        g1d[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));   // Gaussian weight
        sum += g1d[i];
    }
    // Normalise so the kernel sums to 1 (energy preservation)
    for (auto& v : g1d) v /= sum;

    // Build 2-D kernel as the outer product of the 1-D kernel with itself
    cv::Mat kernel(k, k, CV_64F);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < k; ++c)
            kernel.at<double>(r, c) = g1d[r] * g1d[c];

    // Reflect-pad, correlate, and clip to uint8
    return ImageUtils::applyKernelReflect(channel, kernel);
}

// ════════════════════════════════════════════════════════════════════
//  Average (Box) Filter
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Apply a uniform-weight box (average / mean) filter.
 *
 * Every pixel in the output is the arithmetic mean of the k×k
 * neighbourhood centred on it.  Uses reflect-padding at borders.
 *
 * @param channel  Single-channel input (CV_8UC1).
 * @param k        Odd kernel side length (≥ 3).
 * @return         Smoothed CV_8UC1 channel.
 */
cv::Mat FilterProcessor::averageFilter(const cv::Mat& channel, int k) {
    int half = k / 2;

    // Reflect-pad so the kernel fits around border pixels
    cv::Mat padded = ImageUtils::padReflect(channel, half, half);

    cv::Mat output(channel.rows, channel.cols, CV_8UC1);
    double area = k * k;   // Number of pixels in the kernel window

    // Slide the kernel window over every pixel
    for (int i = 0; i < channel.rows; ++i) {
        for (int j = 0; j < channel.cols; ++j) {
            double sum = 0;
            // Accumulate pixel values within the k×k window
            for (int m = 0; m < k; ++m)
                for (int n = 0; n < k; ++n)
                    sum += padded.at<uchar>(i + m, j + n);
            // Compute the mean and truncate to uchar
            output.at<uchar>(i, j) = static_cast<uchar>(sum / area);
        }
    }
    return output;
}

// ════════════════════════════════════════════════════════════════════
//  Median Filter
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Apply a non-linear median filter to a single channel.
 *
 * Replaces each pixel with the median value of its k×k neighbourhood.
 * Particularly effective against salt-and-pepper (impulse) noise
 * because extreme outlier values are discarded.
 *
 * Uses std::nth_element for O(N) partial sorting of the window.
 *
 * @param channel  Single-channel input (CV_8UC1).
 * @param k        Odd kernel side length (≥ 3).
 * @return         Filtered CV_8UC1 channel.
 */
cv::Mat FilterProcessor::medianFilter(const cv::Mat& channel, int k) {
    int half = k / 2;

    // Reflect-pad the input
    cv::Mat padded = ImageUtils::padReflect(channel, half, half);

    cv::Mat output(channel.rows, channel.cols, CV_8UC1);
    std::vector<uchar> window(k * k);   // Re-usable buffer for neighbourhood values

    // Slide the window over every pixel
    for (int i = 0; i < channel.rows; ++i) {
        for (int j = 0; j < channel.cols; ++j) {
            // Collect all values in the k×k neighbourhood
            int idx = 0;
            for (int m = 0; m < k; ++m)
                for (int n = 0; n < k; ++n)
                    window[idx++] = padded.at<uchar>(i + m, j + n);

            // Partial sort to find the median (middle element)
            std::nth_element(window.begin(),
                             window.begin() + window.size() / 2,
                             window.end());
            output.at<uchar>(i, j) = window[window.size() / 2];
        }
    }
    return output;
}
