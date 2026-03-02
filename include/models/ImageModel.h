/**
 * @file ImageModel.h
 * @brief Declaration of the ImageModel class — the application's
 *        data layer for loaded images and cached processing results.
 *
 * Mirrors the Python implementation in Models/Image_Model.py.
 * Stores the original loaded image and provides a key-based cache
 * so that expensive processing results can be reused when the same
 * operation is requested again.
 */

#ifndef IMAGEMODEL_H
#define IMAGEMODEL_H

#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <functional>

/**
 * @class ImageModel
 * @brief Stores a loaded image and caches derived processing results.
 *
 * The model owns the original image (in BGR format, as loaded by
 * OpenCV) and maintains a string-keyed cache of cv::Mat results.
 * Callers can retrieve a cached result or compute-and-cache it in
 * a single call via getOrCompute().
 *
 * Cache entries should be invalidated when the source image changes
 * or when the user explicitly requests a fresh computation.
 */
class ImageModel {
public:
    /** @brief Default constructor — no image loaded yet. */
    ImageModel() = default;

    /**
     * @brief Load an image from disk.
     *
     * The image is stored internally as BGR (OpenCV's default colour
     * order).  Any previously cached results are cleared.
     *
     * @param path  File-system path to the image file.
     * @throws std::runtime_error  If the file cannot be read.
     */
    void load(const std::string& path);

    /**
     * @brief Return a deep copy of the original loaded image.
     *
     * A clone is returned so callers cannot accidentally modify the
     * stored original.
     *
     * @return  CV_8UC3 BGR image.
     * @throws std::runtime_error  If no image has been loaded yet.
     */
    cv::Mat getOriginal() const;

    /**
     * @brief Retrieve a cached result, or compute and cache it.
     *
     * If the key already exists in the cache, a clone is returned.
     * Otherwise `func` is called with the original image, the result
     * is stored under `key`, and a clone is returned.
     *
     * @param key   Unique identifier for this processing result
     *              (e.g. "edge_sobel_combined").
     * @param func  Callable that takes a const cv::Mat& (the original
     *              image) and returns the processed cv::Mat.
     * @return      Deep copy of the cached (or freshly computed) result.
     */
    cv::Mat getOrCompute(const std::string& key,
                         std::function<cv::Mat(const cv::Mat&)> func);

    /**
     * @brief Remove a single entry from the cache.
     * @param key  The cache key to remove.  No-op if the key is absent.
     */
    void invalidate(const std::string& key);

    /**
     * @brief Drop the entire results cache.
     *
     * Useful when the source image changes (new image loaded) so
     * that stale results are not served.
     */
    void clearCache();

    /**
     * @brief Check whether an image has been loaded.
     * @return true if an image is loaded and ready for processing.
     */
    bool isLoaded() const { return !original_.empty(); }

private:
    cv::Mat original_;                       ///< The original loaded image (BGR, CV_8UC3).
    std::map<std::string, cv::Mat> cache_;   ///< Key → processed-result cache.
};

#endif // IMAGEMODEL_H
