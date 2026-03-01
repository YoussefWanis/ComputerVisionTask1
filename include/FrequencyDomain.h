#ifndef FREQUENCYDOMAIN_H
#define FREQUENCYDOMAIN_H

#include <opencv2/opencv.hpp>

struct FFTData {
    cv::Mat complex;      // DFT result (complex)
    cv::Size originalSize; // size of input grayscale image
    cv::Size paddedSize;   // size after padding
};

class FrequencyDomain {
public:
    // Compute FFT of a grayscale image (padded to optimal size)
    static FFTData computeFFT(const cv::Mat& gray);

    // Apply low-pass mask to FFT and return spatial result
    static cv::Mat applyLowPass(const FFTData& fft, float cutoff);

    // Apply high-pass mask to FFT and return spatial result
    static cv::Mat applyHighPass(const FFTData& fft, float cutoff);

    // Legacy methods (still work, but compute FFT each time)
    static cv::Mat lowPass(const cv::Mat& input, float cutoff = 30);
    static cv::Mat highPass(const cv::Mat& input, float cutoff = 30);

private:
    FrequencyDomain() = delete;
};

#endif // FREQUENCYDOMAIN_H