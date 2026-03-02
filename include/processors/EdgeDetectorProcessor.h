/**
 * @file EdgeDetectorProcessor.h
 * @brief Declaration of the EdgeDetectorProcessor class for
 *        gradient-based and Canny edge detection.
 *
 * Mirrors the Python implementation in Processors/edge.py.
 *
 * Supported edge-detection methods:
 *   - "sobel"   — 3×3 Sobel gradient kernels.
 *   - "roberts" — 2×2 Roberts Cross gradient kernels.
 *   - "prewitt" — 3×3 Prewitt gradient kernels.
 *   - "canny"   — OpenCV's built-in Canny algorithm.
 *
 * For gradient methods, the direction can be:
 *   - "x"        — horizontal gradient only.
 *   - "y"        — vertical gradient only.
 *   - "combined" — Euclidean magnitude sqrt(Gx² + Gy²).
 */

#ifndef EDGEDETECTORPROCESSOR_H
#define EDGEDETECTORPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include "utils/ImageUtils.h"

/**
 * @class EdgeDetectorProcessor
 * @brief Detects edges using gradient-based operators (Sobel, Roberts,
 *        Prewitt) or the Canny algorithm.
 *
 * Usage:
 * @code
 *   EdgeDetectorProcessor ep;
 *   cv::Mat edges = ep.process(image, "sobel", "combined");
 *   cv::Mat canny = ep.process(image, "canny", "combined", 50, 150);
 * @endcode
 */
class EdgeDetectorProcessor {
public:
    /** @brief Default constructor. */
    EdgeDetectorProcessor() = default;

    /**
     * @brief Detect edges in the image using the specified method.
     *
     * Routes to either the gradient-based `detect()` or `cannyDetect()`
     * depending on the method string.
     *
     * @param image      Input image (BGR or grayscale). Must not be empty.
     * @param method     Algorithm: "sobel" | "roberts" | "prewitt" | "canny".
     * @param direction  Gradient axis (ignored for Canny):
     *                   "x" | "y" | "combined".
     * @param cannyLow   Lower hysteresis threshold (Canny only, default 50).
     * @param cannyHigh  Upper hysteresis threshold (Canny only, default 150).
     * @return           CV_8UC1 edge-magnitude (or binary edge) image.
     *
     * @throws std::invalid_argument  If the image is empty or an
     *         unknown method / direction is provided.
     */
    cv::Mat process(const cv::Mat& image,
                    const std::string& method    = "sobel",
                    const std::string& direction = "combined",
                    int cannyLow  = 50,
                    int cannyHigh = 150);

private:
    /**
     * @brief Gradient-based edge detection (Sobel / Roberts / Prewitt).
     * @param image      Input image (converted to grayscale internally).
     * @param method     Kernel set: "sobel" | "roberts" | "prewitt".
     * @param direction  "x", "y", or "combined".
     * @return           CV_8UC1 edge-magnitude image.
     */
    cv::Mat detect     (const cv::Mat& image, const std::string& method,
                        const std::string& direction);

    /**
     * @brief Canny edge detection (delegates to cv::Canny).
     * @param image  Input image (converted to grayscale internally).
     * @param low    Lower hysteresis threshold.
     * @param high   Upper hysteresis threshold.
     * @return       CV_8UC1 binary edge map (0 or 255).
     */
    cv::Mat cannyDetect(const cv::Mat& image, int low, int high);
};

#endif // EDGEDETECTORPROCESSOR_H
