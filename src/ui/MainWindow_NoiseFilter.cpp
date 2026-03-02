#include "ui/MainWindow.h"
#include <QMessageBox>
#include <sstream>

// ═══════════════════════════════════════════════════════════════════
//  Noise-type visibility toggle  (matches Python _on_nf_noise_type)
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onNoiseTypeChanged(const QString& type) {
    bool gaus = (type == "gaussian");
    bool sp   = (type == "salt_pepper");

    nfLblMean_->setVisible(gaus);
    nfMean_->setVisible(gaus);
    nfLblStd_->setVisible(gaus);
    nfStd_->setVisible(gaus);
    nfLblRatio_->setVisible(sp);
    nfSPRatio_->setVisible(sp);
}

// ═══════════════════════════════════════════════════════════════════
//  Build noise params (matching Python _build_noise_kwargs)
// ═══════════════════════════════════════════════════════════════════

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

void MainWindow::onApplyNoise() {
    cv::Mat original = requireImage();
    if (original.empty()) return;

    try {
        std::string ntype = nfNoiseType_->currentText().toStdString();
        double intensity  = nfIntensity_->value();
        double mean       = nfMean_->value();
        double stddev     = nfStd_->value();
        double spRatio    = nfSPRatio_->value();

        std::string key = buildNoiseKey(ntype, intensity, mean, stddev, spRatio);

        // Capture copies for lambda
        auto& np = noiseProc_;
        cv::Mat noisy = model_.getOrCompute(key,
            [&np, ntype, intensity, mean, stddev, spRatio](const cv::Mat& img) {
                return np.process(img, ntype, intensity, mean, stddev, spRatio);
            });

        cachedNoisy_ = noisy;

        // Display: original | noisy
        showImageOnLabel(nfImgs_[0], original);
        nfTitles_[0]->setText("Original");
        showImageOnLabel(nfImgs_[1], noisy);
        nfTitles_[1]->setText(QString("Noisy (%1, i=%2)")
                                  .arg(QString::fromStdString(ntype))
                                  .arg(intensity, 0, 'f', 2));
        showImageOnLabel(nfImgs_[2], cv::Mat());
        nfTitles_[2]->clear();

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

void MainWindow::onApplyFilter() {
    cv::Mat original = requireImage();
    if (original.empty()) return;

    // Ensure we have a noisy image first
    if (cachedNoisy_.empty()) {
        onApplyNoise();
        if (cachedNoisy_.empty()) return;
    }

    try {
        std::string ftype = nfFilterType_->currentText().toStdString();
        int kernel = nfKernel_->value();
        if (kernel % 2 == 0) { kernel++; nfKernel_->setValue(kernel); }

        std::string ntype = nfNoiseType_->currentText().toStdString();
        double intensity  = nfIntensity_->value();

        // Cache key includes both noise and filter params
        std::string noiseKey = buildNoiseKey(ntype, intensity,
                                             nfMean_->value(),
                                             nfStd_->value(),
                                             nfSPRatio_->value());
        std::string filtKey = "filtered_" + ftype + "_" +
                              std::to_string(kernel) + "_" + noiseKey;

        cv::Mat noisy = cachedNoisy_;
        auto& fp = filterProc_;
        cv::Mat filtered = model_.getOrCompute(filtKey,
            [&fp, &noisy, ftype, kernel](const cv::Mat& /*img*/) {
                return fp.process(noisy, ftype, kernel);
            });

        // Display: original | noisy | filtered
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
