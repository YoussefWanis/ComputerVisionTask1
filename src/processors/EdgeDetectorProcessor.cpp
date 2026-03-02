/**
 * @file EdgeDetectorProcessor.cpp
 * @brief Implements gradient-based edge detection (Sobel, Roberts, Prewitt)
 *        and Canny edge detection.
 *
 * Mirrors the Python code in Processors/edge.py.  Each gradient method
 * uses a pair of fixed convolution kernels (one for the X direction, one
 * for Y) and produces an edge-magnitude image.  The Canny method
 * delegates to OpenCV's built-in cv::Canny.
 *
 * Supported methods  : "sobel", "roberts", "prewitt", "canny"
 * Supported directions (gradient methods only): "x", "y", "combined"
 */

#include "processors/EdgeDetectorProcessor.h"
#include "processors/ColorProcessor.h"
#include "utils/ImageUtils.h"
#include <stdexcept>
#include <cmath>

// ════════════════════════════════════════════════════════════════════
//  Kernel definitions  (same as Python's KERNELS dict)
// ════════════════════════════════════════════════════════════════════

/**
 * @name Sobel Kernels
 * 3×3 kernels that approximate the first derivative of the image
 * intensity in the X and Y directions, with Gaussian smoothing.
 * @{
 */
static const cv::Mat SOBEL_X  = (cv::Mat_<double>(3,3) <<
    -1, 0, 1,  -2, 0, 2,  -1, 0, 1);
static const cv::Mat SOBEL_Y  = (cv::Mat_<double>(3,3) <<
    -1,-2,-1,   0, 0, 0,   1, 2, 1);
/** @} */

/**
 * @name Roberts Cross Kernels
 * 2×2 kernels that compute the gradient along the two diagonal
 * directions.  Smallest possible edge-detection kernels.
 * @{
 */
static const cv::Mat ROBERTS_X = (cv::Mat_<double>(2,2) <<
     1, 0,   0,-1);
static const cv::Mat ROBERTS_Y = (cv::Mat_<double>(2,2) <<
     0, 1,  -1, 0);
/** @} */

/**
 * @name Prewitt Kernels
 * 3×3 kernels similar to Sobel but without the extra centre weighting,
 * giving uniform averaging along the perpendicular direction.
 * @{
 */
static const cv::Mat PREWITT_X = (cv::Mat_<double>(3,3) <<
    -1, 0, 1,  -1, 0, 1,  -1, 0, 1);
static const cv::Mat PREWITT_Y = (cv::Mat_<double>(3,3) <<
    -1,-1,-1,   0, 0, 0,   1, 1, 1);
/** @} */

// ════════════════════════════════════════════════════════════════════
//  Public API
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Main entry point for edge detection.
 *
 * Routes the request to either the gradient-based `detect()` method or
 * the `cannyDetect()` method depending on the chosen algorithm.
 *
 * @param image      Input image (BGR or grayscale). Must not be empty.
 * @param method     Edge-detection algorithm:
 *                   "sobel" | "roberts" | "prewitt" | "canny".
 * @param direction  Gradient direction (ignored for Canny):
 *                   "x" | "y" | "combined".
 * @param cannyLow   Lower hysteresis threshold for Canny (default 50).
 * @param cannyHigh  Upper hysteresis threshold for Canny (default 150).
 * @return           CV_8UC1 edge-magnitude image (same size as input).
 *
 * @throws std::invalid_argument  If the image is empty, or if an
 *                                unknown method/direction is provided.
 */
cv::Mat EdgeDetectorProcessor::process(const cv::Mat& image,
                                       const std::string& method,
                                       const std::string& direction,
                                       int cannyLow, int cannyHigh) {
    ImageUtils::assertNotEmpty(image, "EdgeDetectorProcessor::process");

    // Canny is handled separately (uses OpenCV's built-in algorithm)
    if (method == "canny")
        return cannyDetect(image, cannyLow, cannyHigh);

    // Validate method and direction strings for gradient-based detectors
    if (method != "sobel" && method != "roberts" && method != "prewitt")
        throw std::invalid_argument("Unknown edge method: " + method);

    if (direction != "x" && direction != "y" && direction != "combined")
        throw std::invalid_argument("Unknown direction: " + direction);

    return detect(image, method, direction);
}

// ════════════════════════════════════════════════════════════════════
//  Gradient-based detection
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Compute edge magnitudes using a pair of gradient kernels.
 *
 * Steps:
 *   1. Convert input to grayscale (if colour).
 *   2. Convert to CV_64F for floating-point arithmetic.
 *   3. Select the appropriate X and Y kernels for the chosen method.
 *   4. Compute the gradient(s) via 2-D cross-correlation.
 *      - "x"        → return |Gx|
 *      - "y"        → return |Gy|
 *      - "combined" → return sqrt(Gx² + Gy²)
 *   5. Convert the result back to CV_8U with saturation.
 *
 * @param image      Input image (any type / channel count).
 * @param method     Kernel set to use: "sobel" | "roberts" | "prewitt".
 * @param direction  Which gradient component(s): "x" | "y" | "combined".
 * @return           CV_8UC1 edge-magnitude image.
 */
cv::Mat EdgeDetectorProcessor::detect(const cv::Mat& image,
                                      const std::string& method,
                                      const std::string& direction) {
    // Step 1: Convert to single-channel grayscale
    cv::Mat gray = ImageUtils::ensureGrayscale(image);

    // Step 2: Promote to 64-bit float for precise convolution
    cv::Mat fgray;
    gray.convertTo(fgray, CV_64F);

    // Step 3: Select the X and Y kernels for the requested method
    const cv::Mat *kx = nullptr, *ky = nullptr;
    if      (method == "sobel")   { kx = &SOBEL_X;   ky = &SOBEL_Y;   }
    else if (method == "roberts") { kx = &ROBERTS_X;  ky = &ROBERTS_Y; }
    else if (method == "prewitt") { kx = &PREWITT_X;  ky = &PREWITT_Y; }

    // Step 4: Correlate with the selected kernel(s)
    cv::Mat G;
    if (direction == "x") {
        // Horizontal gradient only
        G = ImageUtils::correlate2d(fgray, *kx);
    } else if (direction == "y") {
        // Vertical gradient only
        G = ImageUtils::correlate2d(fgray, *ky);
    } else {  // "combined"
        // Compute both gradients and take the Euclidean magnitude
        cv::Mat Gx = ImageUtils::correlate2d(fgray, *kx);
        cv::Mat Gy = ImageUtils::correlate2d(fgray, *ky);
        cv::magnitude(Gx, Gy, G);   // G = sqrt(Gx² + Gy²)
    }

    // Step 5: Convert back to 8-bit unsigned (saturate_cast clips to [0,255])
    cv::Mat out;
    G.convertTo(out, CV_8U);
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  Canny Edge Detection
// ════════════════════════════════════════════════════════════════════

/**
 * @brief Apply the Canny edge detector to the image.
 *
 * Delegates to OpenCV's cv::Canny, which internally performs:
 *   1. Gaussian blur to reduce noise.
 *   2. Sobel gradient computation.
 *   3. Non-maximum suppression.
 *   4. Double-threshold hysteresis edge tracking.
 *
 * @param image  Input image (BGR or grayscale). Converted to grayscale
 *               if needed.
 * @param low    Lower hysteresis threshold — edges with gradient below
 *               this are discarded.
 * @param high   Upper hysteresis threshold — edges with gradient above
 *               this are always kept; those between low and high are
 *               kept only if connected to a strong edge.
 * @return       CV_8UC1 binary edge map (0 or 255).
 */
cv::Mat EdgeDetectorProcessor::cannyDetect(const cv::Mat& image, int low, int high) {
    // Convert to grayscale (Canny requires single-channel input)
    cv::Mat gray = ImageUtils::ensureGrayscale(image);

    cv::Mat edges;
    cv::Canny(gray, edges, low, high);
    return edges;
}
