#include "processors/HistogramProcessor.h"
#include "processors/ColorProcessor.h"
#include "utils/ImageUtils.h"
#include <cmath>
#include <algorithm>

// ════════════════════════════════════════════════════════════════════
//  Public — equalize
// ════════════════════════════════════════════════════════════════════

cv::Mat HistogramProcessor::equalize(const cv::Mat& image) {
    ImageUtils::assertNotEmpty(image, "HistogramProcessor::equalize");
    if (image.channels() == 1)
        return equalizeChannel(image);
    return equalizeBGR(image);
}

// ════════════════════════════════════════════════════════════════════
//  Public — normalize  [min-max → 0..255]
// ════════════════════════════════════════════════════════════════════

cv::Mat HistogramProcessor::normalize(const cv::Mat& image) {
    double mn, mx;
    cv::minMaxLoc(image.reshape(1), &mn, &mx);   // flatten channels
    if (mn == mx) return image.clone();

    cv::Mat fimg;
    image.convertTo(fimg, CV_64F);
    fimg = (fimg - mn) / (mx - mn) * 255.0;

    cv::Mat out;
    fimg.convertTo(out, CV_8U);
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  Public — computeChannelHistograms
//  Returns keys "B", "G", "R" (matching Python's "R", "G", "B"
//  but adapted to BGR channel order).
// ════════════════════════════════════════════════════════════════════

std::map<std::string, ChannelHistData>
HistogramProcessor::computeChannelHistograms(const cv::Mat& bgr) {
    std::map<std::string, ChannelHistData> result;
    std::vector<cv::Mat> channels = ColorProcessor::splitChannels(bgr);
    const char* names[] = {"B", "G", "R"};

    for (int c = 0; c < 3; ++c) {
        ChannelHistData data;
        data.histogram.assign(256, 0);

        const cv::Mat& ch = channels[c];
        for (int i = 0; i < ch.rows; ++i) {
            const uchar* row = ch.ptr<uchar>(i);
            for (int j = 0; j < ch.cols; ++j)
                data.histogram[row[j]]++;
        }

        // cumulative → normalised CDF
        data.cdf.resize(256);
        double cumSum = 0;
        double totalPixels = ch.rows * ch.cols;
        for (int k = 0; k < 256; ++k) {
            cumSum += data.histogram[k];
            data.cdf[k] = cumSum / totalPixels;
        }

        result[names[c]] = std::move(data);
    }
    return result;
}

// ════════════════════════════════════════════════════════════════════
//  Private — single-channel equalisation
// ════════════════════════════════════════════════════════════════════

cv::Mat HistogramProcessor::equalizeChannel(const cv::Mat& channel) {
    // Histogram
    int hist[256] = {};
    for (int i = 0; i < channel.rows; ++i) {
        const uchar* row = channel.ptr<uchar>(i);
        for (int j = 0; j < channel.cols; ++j)
            hist[row[j]]++;
    }

    // CDF
    double cdf[256];
    cdf[0] = hist[0];
    for (int k = 1; k < 256; ++k)
        cdf[k] = cdf[k - 1] + hist[k];

    // cdf_min
    double cdfMin = 0;
    for (int k = 0; k < 256; ++k) {
        if (cdf[k] > 0) { cdfMin = cdf[k]; break; }
    }

    int N = channel.rows * channel.cols;
    double denom = N - cdfMin;
    if (denom == 0) return channel.clone();

    // LUT
    uchar lut[256];
    for (int k = 0; k < 256; ++k)
        lut[k] = cv::saturate_cast<uchar>(
                    std::round((cdf[k] - cdfMin) / denom * 255.0));

    // Apply
    cv::Mat out(channel.size(), CV_8UC1);
    for (int i = 0; i < channel.rows; ++i) {
        const uchar* src = channel.ptr<uchar>(i);
        uchar* dst = out.ptr<uchar>(i);
        for (int j = 0; j < channel.cols; ++j)
            dst[j] = lut[src[j]];
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  Private — BGR equalisation (luminance-based, matching Python)
//  Equalise luminance, then scale each channel proportionally.
// ════════════════════════════════════════════════════════════════════

cv::Mat HistogramProcessor::equalizeBGR(const cv::Mat& bgr) {
    // Luminance
    cv::Mat L = ColorProcessor::toGrayscale(bgr);
    cv::Mat Leq = equalizeChannel(L);

    cv::Mat Lf, Leqf;
    L.convertTo(Lf, CV_64F);
    Leq.convertTo(Leqf, CV_64F);

    const double eps = 1e-8;

    // Scale each channel:  C_out = C * (L_eq / (L + eps))
    cv::Mat result(bgr.size(), bgr.type());
    for (int i = 0; i < bgr.rows; ++i) {
        const uchar* src = bgr.ptr<uchar>(i);
        uchar* dst = result.ptr<uchar>(i);
        const double* lp  = Lf.ptr<double>(i);
        const double* leq = Leqf.ptr<double>(i);
        for (int j = 0; j < bgr.cols; ++j) {
            double scale = leq[j] / (lp[j] + eps);
            for (int c = 0; c < 3; ++c) {
                double val = src[j * 3 + c] * scale;
                dst[j * 3 + c] = cv::saturate_cast<uchar>(val);
            }
        }
    }
    return result;
}
