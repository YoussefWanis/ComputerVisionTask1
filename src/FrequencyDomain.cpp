#include "FrequencyDomain.h"
#include "Utils.h"
#include <cmath>

FFTData FrequencyDomain::computeFFT(const cv::Mat& gray) {
    FFTData data;
    data.originalSize = gray.size();

    // Pad to optimal size
    int m = cv::getOptimalDFTSize(gray.rows);
    int n = cv::getOptimalDFTSize(gray.cols);
    data.paddedSize = cv::Size(n, m);

    cv::Mat padded;
    cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::merge(planes, 2, data.complex);
    cv::dft(data.complex, data.complex);
    return data;
}

static cv::Mat applyMask(const FFTData& fft, const cv::Mat& realMask) {
    // realMask: single‑channel float, same size as padded, values 0/1, already shifted.
    // Convert to two‑channel complex mask (real = realMask, imag = 0)
    cv::Mat maskPlanes[] = {realMask, cv::Mat::zeros(realMask.size(), CV_32F)};
    cv::Mat maskComplex;
    cv::merge(maskPlanes, 2, maskComplex);

    cv::Mat filtered;
    cv::mulSpectrums(fft.complex, maskComplex, filtered, cv::DFT_ROWS);

    cv::Mat inverse;
    cv::dft(filtered, inverse, cv::DFT_INVERSE | cv::DFT_SCALE);

    std::vector<cv::Mat> planes;
    cv::split(inverse, planes);
    cv::Mat result;
    cv::normalize(planes[0], result, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Crop to original size
    return result(cv::Rect(0, 0, fft.originalSize.width, fft.originalSize.height)).clone();
}

cv::Mat FrequencyDomain::applyLowPass(const FFTData& fft, float cutoff) {
    // Create low‑pass mask (circle)
    cv::Mat mask = cv::Mat::zeros(fft.paddedSize, CV_32F);
    int cx = fft.paddedSize.width / 2;
    int cy = fft.paddedSize.height / 2;
    for (int i = 0; i < fft.paddedSize.height; ++i) {
        for (int j = 0; j < fft.paddedSize.width; ++j) {
            float dist = std::sqrt((i - cy)*(i - cy) + (j - cx)*(j - cx));
            if (dist <= cutoff)
                mask.at<float>(i, j) = 1.0f;
        }
    }
    // Shift mask (swap quadrants) to match DFT layout (origin at top‑left)
    int cx2 = fft.paddedSize.width / 2;
    int cy2 = fft.paddedSize.height / 2;
    cv::Mat q0(mask, cv::Rect(0, 0, cx2, cy2));
    cv::Mat q1(mask, cv::Rect(cx2, 0, cx2, cy2));
    cv::Mat q2(mask, cv::Rect(0, cy2, cx2, cy2));
    cv::Mat q3(mask, cv::Rect(cx2, cy2, cx2, cy2));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    return applyMask(fft, mask);
}

cv::Mat FrequencyDomain::applyHighPass(const FFTData& fft, float cutoff) {
    cv::Mat mask = cv::Mat::ones(fft.paddedSize, CV_32F);
    int cx = fft.paddedSize.width / 2;
    int cy = fft.paddedSize.height / 2;
    for (int i = 0; i < fft.paddedSize.height; ++i) {
        for (int j = 0; j < fft.paddedSize.width; ++j) {
            float dist = std::sqrt((i - cy)*(i - cy) + (j - cx)*(j - cx));
            if (dist <= cutoff)
                mask.at<float>(i, j) = 0.0f;
        }
    }
    // Shift mask
    int cx2 = fft.paddedSize.width / 2;
    int cy2 = fft.paddedSize.height / 2;
    cv::Mat q0(mask, cv::Rect(0, 0, cx2, cy2));
    cv::Mat q1(mask, cv::Rect(cx2, 0, cx2, cy2));
    cv::Mat q2(mask, cv::Rect(0, cy2, cx2, cy2));
    cv::Mat q3(mask, cv::Rect(cx2, cy2, cx2, cy2));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    return applyMask(fft, mask);
}

cv::Mat FrequencyDomain::lowPass(const cv::Mat& input, float cutoff) {
    cv::Mat gray = Utils::toGrayscale(input);
    FFTData fft = computeFFT(gray);
    return applyLowPass(fft, cutoff);
}

cv::Mat FrequencyDomain::highPass(const cv::Mat& input, float cutoff) {
    cv::Mat gray = Utils::toGrayscale(input);
    FFTData fft = computeFFT(gray);
    return applyHighPass(fft, cutoff);
}