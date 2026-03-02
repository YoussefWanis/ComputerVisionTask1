#include "processors/HybridProcessor.h"
#include "processors/FFTProcessor.h"
#include "utils/ImageUtils.h"
#include <algorithm>

// ════════════════════════════════════════════════════════════════════
//  create — matches Python's create_hybrid_image()
//
//  hybrid = lowpass(lpImage) + ( hpImage − lowpass(hpImage) )
//  Uses raw (unnormalised) FFT output so the zero-centred
//  high-pass can be added correctly.
// ════════════════════════════════════════════════════════════════════

cv::Mat HybridProcessor::create(const cv::Mat& lpImage,
                                const cv::Mat& hpImage,
                                int cutoffLow,
                                int cutoffHigh) {
    ImageUtils::assertNotEmpty(lpImage, "HybridProcessor LP image");
    ImageUtils::assertNotEmpty(hpImage, "HybridProcessor HP image");
    FFTProcessor fft;

    // Low-pass of the LP source
    cv::Mat lowA = fft.processRaw(lpImage, "lowpass", cutoffLow);

    // High-pass of the HP source = B − lowpass(B)
    cv::Mat lowB = fft.processRaw(hpImage, "lowpass", cutoffHigh);
    cv::Mat hpB;
    hpImage.convertTo(hpB, CV_64F);
    hpB = hpB - lowB;

    // Combine and clip
    cv::Mat hybrid = lowA + hpB;

    // Clip to [0, 255] and convert
    cv::Mat out;
    hybrid.convertTo(out, CV_8U);   // saturate_cast clips
    return out;
}
