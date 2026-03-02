#include "processors/FFTProcessor.h"
#include "utils/ImageUtils.h"
#include <cmath>
#include <stdexcept>

// ════════════════════════════════════════════════════════════════════
//  Public — process (normalised uint8 for display)
// ════════════════════════════════════════════════════════════════════

cv::Mat FFTProcessor::process(const cv::Mat& image,
                              const std::string& filterType,
                              int cutoffRadius) {
    ImageUtils::assertNotEmpty(image, "FFTProcessor::process");
    if (image.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        for (auto& ch : channels)
            ch = filterChannel(ch, filterType, cutoffRadius);
        cv::Mat result;
        cv::merge(channels, result);
        return result;
    }
    return filterChannel(image, filterType, cutoffRadius);
}

// ════════════════════════════════════════════════════════════════════
//  Public — processRaw (float64, for hybrid images)
// ════════════════════════════════════════════════════════════════════

cv::Mat FFTProcessor::processRaw(const cv::Mat& image,
                                 const std::string& filterType,
                                 int cutoffRadius) {
    ImageUtils::assertNotEmpty(image, "FFTProcessor::processRaw");
    if (image.channels() == 3) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        for (auto& ch : channels)
            ch = filterChannelRaw(ch, filterType, cutoffRadius);
        cv::Mat result;
        cv::merge(channels, result);
        return result;
    }
    return filterChannelRaw(image, filterType, cutoffRadius);
}

// ════════════════════════════════════════════════════════════════════
//  Circular mask  (matching Python's _make_circular_mask)
// ════════════════════════════════════════════════════════════════════

cv::Mat FFTProcessor::makeCircularMask(cv::Size shape,
                                       const std::string& filterType,
                                       int cutoffRadius) {
    int H = shape.height, W = shape.width;
    int cy = H / 2, cx = W / 2;

    cv::Mat mask(H, W, CV_64F);

    for (int i = 0; i < H; ++i) {
        double* row = mask.ptr<double>(i);
        for (int j = 0; j < W; ++j) {
            double dist = std::sqrt(static_cast<double>((j - cx) * (j - cx)
                                  + (i - cy) * (i - cy)));
            if (filterType == "lowpass")
                row[j] = (dist <= cutoffRadius) ? 1.0 : 0.0;
            else  // highpass
                row[j] = (dist > cutoffRadius) ? 1.0 : 0.0;
        }
    }
    return mask;
}

// ════════════════════════════════════════════════════════════════════
//  Core FFT pipeline — raw float64 output
//
//  Steps (matching Python exactly):
//    1. fft2
//    2. fftshift       → move DC to centre
//    3. multiply mask
//    4. ifftshift       (same as fftshift for even dims)
//    5. ifft2
//    6. abs
// ════════════════════════════════════════════════════════════════════

cv::Mat FFTProcessor::filterChannelRaw(const cv::Mat& channel,
                                       const std::string& filterType,
                                       int cutoffRadius) {
    cv::Mat floatCh;
    channel.convertTo(floatCh, CV_64F);

    // Build complex input (2-channel: real + imag=0)
    cv::Mat planes[] = {floatCh,
                        cv::Mat::zeros(floatCh.size(), CV_64F)};
    cv::Mat complex;
    cv::merge(planes, 2, complex);

    // 1. DFT
    cv::dft(complex, complex);

    // 2. fftshift
    ImageUtils::fftShift(complex);

    // 3. Apply circular mask (multiply both real & imag parts)
    cv::Mat mask = makeCircularMask(channel.size(), filterType, cutoffRadius);
    cv::split(complex, planes);
    cv::multiply(planes[0], mask, planes[0]);
    cv::multiply(planes[1], mask, planes[1]);
    cv::merge(planes, 2, complex);

    // 4. ifftshift (= fftshift for even dimensions)
    ImageUtils::fftShift(complex);

    // 5. Inverse DFT
    cv::dft(complex, complex, cv::DFT_INVERSE | cv::DFT_SCALE);

    // 6. Magnitude = abs of complex result
    cv::split(complex, planes);
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);

    return magnitude;
}

// ════════════════════════════════════════════════════════════════════
//  Normalised filter — [0,255] uint8 for display
// ════════════════════════════════════════════════════════════════════

cv::Mat FFTProcessor::filterChannel(const cv::Mat& channel,
                                    const std::string& filterType,
                                    int cutoffRadius) {
    cv::Mat raw = filterChannelRaw(channel, filterType, cutoffRadius);

    double mn, mx;
    cv::minMaxLoc(raw, &mn, &mx);
    if (mx > mn)
        raw = (raw - mn) / (mx - mn) * 255.0;

    cv::Mat out;
    raw.convertTo(out, CV_8U);
    return out;
}

