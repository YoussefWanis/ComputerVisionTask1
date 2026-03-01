#include "Hybrid.h"
#include "FrequencyDomain.h"
#include "Utils.h"

cv::Mat Hybrid::create(const cv::Mat& img1, const cv::Mat& img2,
                       float cutoff1, float cutoff2) {
    cv::Mat low = FrequencyDomain::lowPass(img1, cutoff1);
    cv::Mat high = FrequencyDomain::highPass(img2, cutoff2);

    // Ensure same size
    if (low.size() != high.size()) {
        cv::resize(high, high, low.size());
    }

    cv::Mat hybrid;
    cv::addWeighted(low, 0.5, high, 0.5, 0, hybrid);
    return hybrid;
}