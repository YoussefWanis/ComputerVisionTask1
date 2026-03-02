#ifndef IMAGEMODEL_H
#define IMAGEMODEL_H

#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <functional>

/**
 * ImageModel — stores the loaded image and a key→cv::Mat cache.
 * Mirrors Python's Models/Image_Model.py.
 */
class ImageModel {
public:
    ImageModel() = default;

    /** Load an image from disk (stored internally as BGR). */
    void load(const std::string& path);

    /** Return a deep copy of the original loaded image. */
    cv::Mat getOriginal() const;

    /** Retrieve a cached result for @p key, or compute+cache it. */
    cv::Mat getOrCompute(const std::string& key,
                         std::function<cv::Mat(const cv::Mat&)> func);

    /** Remove one cached entry. */
    void invalidate(const std::string& key);

    /** Drop the entire cache. */
    void clearCache();

    bool isLoaded() const { return !original_.empty(); }

private:
    cv::Mat original_;
    std::map<std::string, cv::Mat> cache_;
};

#endif // IMAGEMODEL_H
