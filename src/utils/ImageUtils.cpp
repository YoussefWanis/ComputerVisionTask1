#include "utils/ImageUtils.h"
#include "processors/ColorProcessor.h"
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════
//  Validation
// ═══════════════════════════════════════════════════════════════════

void ImageUtils::assertNotEmpty(const cv::Mat& img,
                                const std::string& context) {
    if (img.empty()) {
        std::string msg = "Image is empty";
        if (!context.empty()) msg += " (" + context + ")";
        throw std::invalid_argument(msg);
    }
}

void ImageUtils::validateKernelSize(int kernelSize) {
    if (kernelSize < 3 || kernelSize % 2 == 0)
        throw std::invalid_argument(
            "Invalid kernel size (must be odd >= 3), got "
            + std::to_string(kernelSize));
}

// ═══════════════════════════════════════════════════════════════════
//  Colour helpers
// ═══════════════════════════════════════════════════════════════════

cv::Mat ImageUtils::ensureGrayscale(const cv::Mat& image) {
    if (image.channels() == 1)
        return image.clone();
    return ColorProcessor::toGrayscale(image);
}

// ═══════════════════════════════════════════════════════════════════
//  Padding
// ═══════════════════════════════════════════════════════════════════

cv::Mat ImageUtils::padReflect(const cv::Mat& channel, int padH, int padW) {
    cv::Mat padded;
    cv::copyMakeBorder(channel, padded,
                       padH, padH, padW, padW,
                       cv::BORDER_REFLECT);
    return padded;
}

cv::Mat ImageUtils::padZero(const cv::Mat& channel, int padH, int padW) {
    cv::Mat padded;
    cv::copyMakeBorder(channel, padded,
                       padH, padH, padW, padW,
                       cv::BORDER_CONSTANT, cv::Scalar(0));
    return padded;
}

// ═══════════════════════════════════════════════════════════════════
//  2-D cross-correlation  (matches Python EdgeDetectorProcessor._convolve2d)
//
//  Input channel must be CV_64F.  The kernel can be any size (even/odd).
//  Uses zero-padding, result has the same dimensions as channel.
// ═══════════════════════════════════════════════════════════════════

cv::Mat ImageUtils::correlate2d(const cv::Mat& channel,
                                const cv::Mat& kernel) {
    int kh = kernel.rows, kw = kernel.cols;
    int padH = kh / 2, padW = kw / 2;
    int H = channel.rows, W = channel.cols;

    cv::Mat padded = padZero(channel, padH, padW);
    cv::Mat result(H, W, CV_64F);

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double sum = 0;
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

cv::Mat ImageUtils::applyKernelReflect(const cv::Mat& channel,
                                       const cv::Mat& kernel) {
    int kh = kernel.rows, kw = kernel.cols;
    int padH = kh / 2, padW = kw / 2;

    cv::Mat padded = padReflect(channel, padH, padW);
    cv::Mat output(channel.rows, channel.cols, CV_8UC1);

    for (int i = 0; i < channel.rows; ++i) {
        for (int j = 0; j < channel.cols; ++j) {
            double val = 0;
            for (int m = 0; m < kh; ++m)
                for (int n = 0; n < kw; ++n)
                    val += padded.at<uchar>(i + m, j + n)
                         * kernel.at<double>(m, n);
            output.at<uchar>(i, j) = cv::saturate_cast<uchar>(val);
        }
    }
    return output;
}

// ═══════════════════════════════════════════════════════════════════
//  fftShift — swap diagonally opposite quadrants (in-place)
// ═══════════════════════════════════════════════════════════════════

void ImageUtils::fftShift(cv::Mat& mat) {
    int cx = mat.cols / 2;
    int cy = mat.rows / 2;

    cv::Mat q0(mat, cv::Rect(0,  0,  cx, cy));   // top-left
    cv::Mat q1(mat, cv::Rect(cx, 0,  cx, cy));   // top-right
    cv::Mat q2(mat, cv::Rect(0,  cy, cx, cy));   // bottom-left
    cv::Mat q3(mat, cv::Rect(cx, cy, cx, cy));   // bottom-right

    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}
