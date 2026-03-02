#include "metrics/MetricsCalculator.h"
#include <cmath>
#include <limits>

double MetricsCalculator::mse(const cv::Mat& original,
                              const cv::Mat& processed) const {
    CV_Assert(original.size() == processed.size());
    CV_Assert(original.type() == processed.type());

    cv::Mat diff;
    original.convertTo(diff, CV_64F);
    cv::Mat proc64;
    processed.convertTo(proc64, CV_64F);

    diff = diff - proc64;
    diff = diff.mul(diff);

    cv::Scalar s = cv::mean(diff);
    // Average over all channels
    double sum = 0;
    for (int c = 0; c < diff.channels(); ++c)
        sum += s[c];
    return sum / diff.channels();
}

double MetricsCalculator::psnr(const cv::Mat& original,
                               const cv::Mat& processed) const {
    double m = mse(original, processed);
    if (m == 0.0)
        return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10((255.0 * 255.0) / m);
}

double MetricsCalculator::snr(const cv::Mat& original,
                              const cv::Mat& processed) const {
    CV_Assert(original.size() == processed.size());
    CV_Assert(original.type() == processed.type());

    cv::Mat orig64, proc64;
    original.convertTo(orig64, CV_64F);
    processed.convertTo(proc64, CV_64F);

    cv::Mat noise = proc64 - orig64;

    cv::Scalar sigPow = cv::mean(orig64.mul(orig64));
    cv::Scalar noiPow = cv::mean(noise.mul(noise));

    double sp = 0, np = 0;
    for (int c = 0; c < orig64.channels(); ++c) {
        sp += sigPow[c];
        np += noiPow[c];
    }
    sp /= orig64.channels();
    np /= orig64.channels();

    if (np == 0.0)
        return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(sp / np);
}

std::map<std::string, double>
MetricsCalculator::computeAll(const cv::Mat& original,
                              const cv::Mat& processed) const {
    return {
        {"MSE",  mse(original, processed)},
        {"PSNR", psnr(original, processed)},
        {"SNR",  snr(original, processed)}
    };
}
