/**
 * @file MainWindow_NoiseFilter.cpp
 * @brief Implements Tab 1 (Noise & Filter) slot handlers for MainWindow.
 *
 * Contains three slots:
 *   - onNoiseTypeChanged() — toggles visibility of noise-type-specific
 *                            parameter controls (Gaussian mean/std,
 *                            salt-pepper ratio).
 *   - onApplyNoise()       — generates a noisy version of the loaded image.
 *   - onApplyFilter()      — applies a spatial filter to the noisy image.
 *
 * Also contains a file-local helper:
 *   - buildNoiseKey()      — creates a unique cache key encoding the
 *                            noise type and all relevant parameters.
 */

#include "ui/MainWindow.h"
#include <QMessageBox>
#include <sstream>

// ═══════════════════════════════════════════════════════════════════
//  Noise-type visibility toggle  (matches Python _on_nf_noise_type)
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: show or hide noise-type-specific parameter controls.
 *
 * - When "gaussian" is selected:  Mean and Std controls are shown.
 * - When "salt_pepper" is selected: S/P Ratio control is shown.
 * - When "uniform" is selected: all extra controls are hidden.
 *
 * @param type  The newly selected noise type string.
 */
void MainWindow::onNoiseTypeChanged(const QString& type) {
    bool gaus = (type == "gaussian");
    bool sp   = (type == "salt_pepper");

    // Show/hide Gaussian-specific controls
    nfLblMean_->setVisible(gaus);
    nfMean_->setVisible(gaus);
    nfLblStd_->setVisible(gaus);
    nfStd_->setVisible(gaus);

    // Show/hide salt-and-pepper specific controls
    nfLblRatio_->setVisible(sp);
    nfSPRatio_->setVisible(sp);
}

// ═══════════════════════════════════════════════════════════════════
//  Build noise cache key  (matching Python _build_noise_kwargs)
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Build a unique cache key string from the noise parameters.
 *
 * The key includes the noise type and intensity, plus any type-specific
 * parameters:
 *   - Gaussian: mean and standard deviation.
 *   - Salt & pepper: salt/pepper ratio.
 *
 * Example keys:
 *   "noisy_gaussian_i0.3_m0_s100"
 *   "noisy_salt_pepper_i0.5_r0.5"
 *
 * @param ntype      Noise type string.
 * @param intensity  Noise intensity.
 * @param mean       Gaussian mean.
 * @param std        Gaussian standard deviation.
 * @param spRatio    Salt fraction for salt_pepper noise.
 * @return           Unique cache key string.
 */
static std::string buildNoiseKey(const std::string& ntype,
                                 double intensity,
                                 double mean, double std,
                                 double spRatio) {
    std::ostringstream oss;
    oss << "noisy_" << ntype << "_i" << intensity;
    if (ntype == "gaussian")
        oss << "_m" << mean << "_s" << std;
    else if (ntype == "salt_pepper")
        oss << "_r" << spRatio;
    return oss.str();
}

// ═══════════════════════════════════════════════════════════════════
//  Apply Noise
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: add noise to the loaded image and display the result.
 *
 * Steps:
 *   1. Read noise parameters from UI controls.
 *   2. Build a unique cache key encoding the noise configuration.
 *   3. Compute (or retrieve from cache) the noisy image via
 *      NoiseProcessor::process.
 *   4. Store the noisy image in cachedNoisy_ so that onApplyFilter()
 *      can operate on it.
 *   5. Display: [Original | Noisy] with quality metrics.
 */
void MainWindow::onApplyNoise() {
    cv::Mat original = requireImage();
    if (original.empty()) return;

    try {
        // Read noise parameters from UI controls
        std::string ntype = nfNoiseType_->currentText().toStdString();
        double intensity  = nfIntensity_->value();
        double mean       = nfMean_->value();
        double stddev     = nfStd_->value();
        double spRatio    = nfSPRatio_->value();

        // Unique cache key for this noise configuration
        std::string key = buildNoiseKey(ntype, intensity, mean, stddev, spRatio);

        // Generate the noisy image (cached to avoid re-randomising)
        auto& np = noiseProc_;
        cv::Mat noisy = model_.getOrCompute(key,
            [&np, ntype, intensity, mean, stddev, spRatio](const cv::Mat& img) {
                return np.process(img, ntype, intensity, mean, stddev, spRatio);
            });

        // Cache the noisy result so the filter can operate on it
        cachedNoisy_ = noisy;

        // ── Update display ──────────────────────────────────────
        showImageOnLabel(nfImgs_[0], original);
        nfTitles_[0]->setText("Original");
        showImageOnLabel(nfImgs_[1], noisy);
        nfTitles_[1]->setText(QString("Noisy (%1, i=%2)")
                                  .arg(QString::fromStdString(ntype))
                                  .arg(intensity, 0, 'f', 2));
        // Clear the filtered panel (not computed yet)
        showImageOnLabel(nfImgs_[2], cv::Mat());
        nfTitles_[2]->clear();

        // Show noise quality metrics; reset filter metrics
        setMetricsText(nfNoiseMetrics_, original, noisy, "Noise — ");
        resetMetrics(nfFilterMetrics_, "Filter — ");

        statusBar()->showMessage(
            QString("Noise: %1, intensity=%2")
                .arg(QString::fromStdString(ntype))
                .arg(intensity, 0, 'f', 2));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Noise Error", e.what());
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Apply Filter  (on the current noisy image)
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: apply a spatial filter to the current noisy image.
 *
 * If no noisy image exists yet, onApplyNoise() is called first to
 * generate one.
 *
 * Steps:
 *   1. Ensure a noisy image is available.
 *   2. Read filter parameters (type, kernel size) from UI controls.
 *   3. Enforce odd kernel size (increment by 1 if even).
 *   4. Build a cache key encoding both noise and filter parameters.
 *   5. Filter the noisy image via FilterProcessor::process (cached).
 *   6. Display: [Original | Noisy | Filtered] with quality metrics.
 */
void MainWindow::onApplyFilter() {
    cv::Mat original = requireImage();
    if (original.empty()) return;

    // Ensure we have a noisy image to filter
    if (cachedNoisy_.empty()) {
        onApplyNoise();                    // Generate noise first
        if (cachedNoisy_.empty()) return;  // Still empty → user cancelled or error
    }

    try {
        // Read filter parameters
        std::string ftype = nfFilterType_->currentText().toStdString();
        int kernel = nfKernel_->value();

        // Enforce odd kernel size (required by all spatial filters)
        if (kernel % 2 == 0) { kernel++; nfKernel_->setValue(kernel); }

        // Build a composite cache key: noise params + filter params
        std::string ntype = nfNoiseType_->currentText().toStdString();
        double intensity  = nfIntensity_->value();
        std::string noiseKey = buildNoiseKey(ntype, intensity,
                                             nfMean_->value(),
                                             nfStd_->value(),
                                             nfSPRatio_->value());
        std::string filtKey = "filtered_" + ftype + "_" +
                              std::to_string(kernel) + "_" + noiseKey;

        // Apply the filter to the noisy image (cached)
        cv::Mat noisy = cachedNoisy_;
        auto& fp = filterProc_;
        cv::Mat filtered = model_.getOrCompute(filtKey,
            [&fp, &noisy, ftype, kernel](const cv::Mat& /*img*/) {
                // NOTE: we filter the noisy image, not the original
                return fp.process(noisy, ftype, kernel);
            });

        // ── Update display: Original | Noisy | Filtered ─────────
        showImageOnLabel(nfImgs_[0], original);
        nfTitles_[0]->setText("Original");
        showImageOnLabel(nfImgs_[1], cachedNoisy_);
        nfTitles_[1]->setText(QString("Noisy (%1, i=%2)")
                                  .arg(QString::fromStdString(ntype))
                                  .arg(intensity, 0, 'f', 2));
        showImageOnLabel(nfImgs_[2], filtered);
        nfTitles_[2]->setText(QString("Filtered (%1, k=%2)")
                                  .arg(QString::fromStdString(ftype))
                                  .arg(kernel));

        // Quality metrics for both noise and filter
        setMetricsText(nfNoiseMetrics_,  original, cachedNoisy_, "Noise — ");
        setMetricsText(nfFilterMetrics_, original, filtered,     "Filter — ");

        statusBar()->showMessage(
            QString("Filter: %1 k=%2 on %3 noise")
                .arg(QString::fromStdString(ftype))
                .arg(kernel)
                .arg(QString::fromStdString(ntype)));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Filter Error", e.what());
    }
}
