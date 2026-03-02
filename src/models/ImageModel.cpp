#include "models/ImageModel.h"
#include <stdexcept>

void ImageModel::load(const std::string& path) {
    cv::Mat raw = cv::imread(path);
    if (raw.empty())
        throw std::runtime_error("Image not found or invalid path: " + path);
    original_ = raw;           // stored as BGR (OpenCV default)
    cache_.clear();
}

cv::Mat ImageModel::getOriginal() const {
    if (original_.empty())
        throw std::runtime_error("No image loaded. Call load() first.");
    return original_.clone();
}

cv::Mat ImageModel::getOrCompute(const std::string& key,
                                 std::function<cv::Mat(const cv::Mat&)> func) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        cv::Mat result = func(getOriginal());
        cache_[key] = result.clone();
        return result.clone();
    }
    return it->second.clone();
}

void ImageModel::invalidate(const std::string& key) {
    cache_.erase(key);
}

void ImageModel::clearCache() {
    cache_.clear();
}
