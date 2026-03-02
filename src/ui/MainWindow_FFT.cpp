/**
 * @file MainWindow_FFT.cpp
 * @brief Implements Tab 4 (Frequency Domain) slot handler for MainWindow.
 *
 * Contains the onApplyFFT() slot which:
 *   1. Applies an ideal circular low-pass or high-pass filter in the
 *      frequency domain via FFTProcessor.
 *   2. Computes and caches the FFT magnitude spectrum for display.
 *   3. Shows three panels: Original | FFT Spectrum | Filtered result.
 *   4. Computes quality metrics comparing the original to the filtered output.
 */

#include "ui/MainWindow.h"
#include "utils/ImageUtils.h"
#include <QMessageBox>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════
//  Apply FFT Filter
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: apply an FFT frequency filter and display the results.
 *
 * Processing steps:
 *   1. Read the filter type ("lowpass" / "highpass") and cutoff radius
 *      from the UI controls.
 *   2. Filter the image via FFTProcessor::process() (cached through
 *      ImageModel::getOrCompute).
 *   3. If the magnitude spectrum has not been computed for this image
 *      yet, compute it:
 *        a. Convert to grayscale.
 *        b. Forward DFT → fftShift → magnitude.
 *        c. Apply log(1 + |F|) for perceptual scaling.
 *        d. Min-max normalise to [0, 255] for display.
 *   4. Update the three display panels:
 *        [0] Original image.
 *        [1] FFT magnitude spectrum (log-scaled).
 *        [2] Filtered result.
 *   5. Compute quality metrics (original vs. filtered).
 */
void MainWindow::onApplyFFT() {
    cv::Mat original = requireImage();
    if (original.empty()) return;

    try {
        // Read user-selected filter parameters
        std::string ftype = fftType_->currentText().toStdString();
        int cutoff = fftCutoff_->value();

        // Cache key encodes filter type and cutoff
        std::string key = "fft_" + ftype + "_" + std::to_string(cutoff);

        // Apply the frequency-domain filter (cached)
        auto& fp = fftProc_;
        cv::Mat result = model_.getOrCompute(key,
            [&fp, ftype, cutoff](const cv::Mat& img) {
                return fp.process(img, ftype, cutoff);
            });

        // ── Compute the magnitude spectrum (once per loaded image) ──
        if (cachedSpectrum_.empty()) {
            // Convert to grayscale for a single-channel spectrum
            cv::Mat gray = (original.channels() == 3)
                           ? ColorProcessor::toGrayscale(original)
                           : original;

            // Convert to 64-bit float for DFT
            cv::Mat fgray;
            gray.convertTo(fgray, CV_64F);

            // Build complex input [real, imag=0] and compute forward DFT
            cv::Mat planes[] = {fgray,
                                cv::Mat::zeros(fgray.size(), CV_64F)};
            cv::Mat complex;
            cv::merge(planes, 2, complex);
            cv::dft(complex, complex);

            // Shift DC to centre for visualisation
            ImageUtils::fftShift(complex);

            // Compute magnitude: |F| = sqrt(real² + imag²)
            cv::split(complex, planes);
            cv::Mat mag;
            cv::magnitude(planes[0], planes[1], mag);

            // Log-scale for perceptual brightness: log(1 + |F|)
            mag += 1.0;
            cv::log(mag, mag);

            // Min-max normalise to [0, 255] for 8-bit display
            double mn, mx;
            cv::minMaxLoc(mag, &mn, &mx);
            if (mx > mn)
                mag = (mag - mn) / (mx - mn) * 255.0;
            mag.convertTo(cachedSpectrum_, CV_8U);
        }

        // ── Update display panels ──────────────────────────────
        showImageOnLabel(fftImgs_[0], original);
        fftTitles_[0]->setText("Original");

        showImageOnLabel(fftImgs_[1], cachedSpectrum_);
        fftTitles_[1]->setText("FFT Spectrum");

        showImageOnLabel(fftImgs_[2], result);
        // Capitalise the filter type for the title (e.g. "Lowpass")
        QString ftTitle = QString::fromStdString(ftype);
        ftTitle[0] = ftTitle[0].toUpper();
        fftTitles_[2]->setText(QString("%1 (r=%2)").arg(ftTitle).arg(cutoff));

        // Compute quality metrics: original vs. filtered
        setMetricsText(fftMetrics_, original, result);

        statusBar()->showMessage(
            QString("FFT %1, cutoff=%2")
                .arg(QString::fromStdString(ftype)).arg(cutoff));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "FFT Error", e.what());
    }
}
