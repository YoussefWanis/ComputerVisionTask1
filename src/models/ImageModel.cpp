#include "models/ImageModel.h"
#include <stdexcept>

/**
 * @file ImageModel.cpp
 * @brief Implements the ImageModel data-layer class.
 *
 * Manages loading images from disk, providing deep copies of the
 * original, and a key-based cache for storing and retrieving
 * expensive processing results.
 *
 * Mirrors the Python implementation in Models/Image_Model.py.
 */

/**
 * @brief Load an image from the specified file path.
 *
 * Uses OpenCV's cv::imread to load the image in grayscale.
 * The existing results cache is cleared because
 * cached results from a previous image are no longer valid.
 *
 * @param path  File-system path to the image file.
 * @throws std::runtime_error  If the image cannot be loaded
 *         (file not found or unsupported format).
 */
void ImageModel::load(const std::string& path) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (raw.empty())
        throw std::runtime_error("Image not found or invalid path: " + path);
    original_ = raw;    // Store the loaded image (Grayscale, CV_8UC1)
    cache_.clear();     // Invalidate all cached results from a previous image
}

/**
 * @brief Return a deep copy of the original loaded image.
 *
 * A clone is returned so that callers can freely modify the result
 * without affecting the stored original.
 *
 * @return  Deep copy of the original image (BGR, CV_8UC3).
 * @throws std::runtime_error  If no image has been loaded yet.
 */
cv::Mat ImageModel::getOriginal() const {
    if (original_.empty())
        throw std::runtime_error("No image loaded. Call load() first.");
    return original_.clone();
}

/**
 * @brief Retrieve a cached processing result, or compute and cache it.
 *
 * If the cache contains an entry for `key`, a clone of that entry is
 * returned immediately (O(1) lookup).  Otherwise `func` is called
 * with the original image, its result is cloned into the cache, and
 * another clone is returned to the caller.
 *
 * Deep copies are used throughout so that neither the caller nor the
 * cache hold shared references that could lead to accidental mutation.
 *
 * @param key   Unique cache key (e.g. "edge_sobel_combined").
 * @param func  Callable: (const cv::Mat& original) ? cv::Mat result.
 * @return      Deep copy of the cached or freshly computed result.
 */
cv::Mat ImageModel::getOrCompute(const std::string& key,
                                 std::function<cv::Mat(const cv::Mat&)> func) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        // Not cached — compute, store, and return a clone
        cv::Mat result = func(getOriginal());
        cache_[key] = result.clone();
        return result.clone();
    }
    // Already cached — return a clone
    return it->second.clone();
}

/**
 * @brief Remove a single entry from the results cache.
 *
 * No-op if the key does not exist.  Useful when a specific result
 * needs to be recomputed (e.g. after parameter changes).
 *
 * @param key  The cache key to remove.
 */
void ImageModel::invalidate(const std::string& key) {
    cache_.erase(key);
}

/**
 * @brief Drop the entire results cache.
 *
 * Called automatically when a new image is loaded to prevent stale
 * results from being served.
 */
void ImageModel::clearCache() {
    cache_.clear();
}
