#include "processors/EdgeDetectorProcessor.h"
#include "processors/ColorProcessor.h"
#include "utils/ImageUtils.h"
#include <stdexcept>
#include <cmath>

// ════════════════════════════════════════════════════════════════════
//  Kernel definitions  (same as Python's KERNELS dict)
// ════════════════════════════════════════════════════════════════════

static const cv::Mat SOBEL_X  = (cv::Mat_<double>(3,3) <<
    -1, 0, 1,  -2, 0, 2,  -1, 0, 1);
static const cv::Mat SOBEL_Y  = (cv::Mat_<double>(3,3) <<
    -1,-2,-1,   0, 0, 0,   1, 2, 1);

static const cv::Mat ROBERTS_X = (cv::Mat_<double>(2,2) <<
     1, 0,   0,-1);
static const cv::Mat ROBERTS_Y = (cv::Mat_<double>(2,2) <<
     0, 1,  -1, 0);

static const cv::Mat PREWITT_X = (cv::Mat_<double>(3,3) <<
    -1, 0, 1,  -1, 0, 1,  -1, 0, 1);
static const cv::Mat PREWITT_Y = (cv::Mat_<double>(3,3) <<
    -1,-1,-1,   0, 0, 0,   1, 1, 1);

// ════════════════════════════════════════════════════════════════════
//  Public API
// ════════════════════════════════════════════════════════════════════

cv::Mat EdgeDetectorProcessor::process(const cv::Mat& image,
                                       const std::string& method,
                                       const std::string& direction,
                                       int cannyLow, int cannyHigh) {
    ImageUtils::assertNotEmpty(image, "EdgeDetectorProcessor::process");
    if (method == "canny")
        return cannyDetect(image, cannyLow, cannyHigh);

    if (method != "sobel" && method != "roberts" && method != "prewitt")
        throw std::invalid_argument("Unknown edge method: " + method);

    if (direction != "x" && direction != "y" && direction != "combined")
        throw std::invalid_argument("Unknown direction: " + direction);

    return detect(image, method, direction);
}

// ════════════════════════════════════════════════════════════════════
//  Gradient-based detection
// ════════════════════════════════════════════════════════════════════

cv::Mat EdgeDetectorProcessor::detect(const cv::Mat& image,
                                      const std::string& method,
                                      const std::string& direction) {
    cv::Mat gray = ImageUtils::ensureGrayscale(image);

    cv::Mat fgray;
    gray.convertTo(fgray, CV_64F);

    // Select kernels
    const cv::Mat *kx = nullptr, *ky = nullptr;
    if      (method == "sobel")   { kx = &SOBEL_X;   ky = &SOBEL_Y;   }
    else if (method == "roberts") { kx = &ROBERTS_X;  ky = &ROBERTS_Y; }
    else if (method == "prewitt") { kx = &PREWITT_X;  ky = &PREWITT_Y; }

    cv::Mat G;
    if (direction == "x") {
        G = ImageUtils::correlate2d(fgray, *kx);
    } else if (direction == "y") {
        G = ImageUtils::correlate2d(fgray, *ky);
    } else {  // combined
        cv::Mat Gx = ImageUtils::correlate2d(fgray, *kx);
        cv::Mat Gy = ImageUtils::correlate2d(fgray, *ky);
        cv::magnitude(Gx, Gy, G);
    }

    // Clip to [0, 255] and convert to uint8
    cv::Mat out;
    G.convertTo(out, CV_8U);   // saturate_cast handles clipping
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  Canny  (uses cv::Canny, matching Python)
// ════════════════════════════════════════════════════════════════════

cv::Mat EdgeDetectorProcessor::cannyDetect(const cv::Mat& image, int low, int high) {
    cv::Mat gray = ImageUtils::ensureGrayscale(image);

    cv::Mat edges;
    cv::Canny(gray, edges, low, high);
    return edges;
}
