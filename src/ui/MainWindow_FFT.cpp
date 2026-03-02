#include "ui/MainWindow.h"
#include "utils/ImageUtils.h"
#include <QMessageBox>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════
//  Apply FFT Filter
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onApplyFFT() {
    cv::Mat original = requireImage();
    if (original.empty()) return;

    try {
        std::string ftype = fftType_->currentText().toStdString();
        int cutoff = fftCutoff_->value();

        std::string key = "fft_" + ftype + "_" + std::to_string(cutoff);

        auto& fp = fftProc_;
        cv::Mat result = model_.getOrCompute(key,
            [&fp, ftype, cutoff](const cv::Mat& img) {
                return fp.process(img, ftype, cutoff);
            });

        // Compute FFT magnitude spectrum (cached once per loaded image)
        if (cachedSpectrum_.empty()) {
            cv::Mat gray = (original.channels() == 3)
                           ? ColorProcessor::toGrayscale(original)
                           : original;

            cv::Mat fgray;
            gray.convertTo(fgray, CV_64F);

            // DFT
            cv::Mat planes[] = {fgray,
                                cv::Mat::zeros(fgray.size(), CV_64F)};
            cv::Mat complex;
            cv::merge(planes, 2, complex);
            cv::dft(complex, complex);

            // fftshift (using shared utility)
            ImageUtils::fftShift(complex);

            // Magnitude spectrum: log(1 + |F|) normalised to [0,255]
            cv::split(complex, planes);
            cv::Mat mag;
            cv::magnitude(planes[0], planes[1], mag);

            mag += 1.0;
            cv::log(mag, mag);

            double mn, mx;
            cv::minMaxLoc(mag, &mn, &mx);
            if (mx > mn)
                mag = (mag - mn) / (mx - mn) * 255.0;
            mag.convertTo(cachedSpectrum_, CV_8U);
        }

        // Display: original | spectrum | filtered
        showImageOnLabel(fftImgs_[0], original);
        fftTitles_[0]->setText("Original");

        showImageOnLabel(fftImgs_[1], cachedSpectrum_);
        fftTitles_[1]->setText("FFT Spectrum");

        showImageOnLabel(fftImgs_[2], result);
        QString ftTitle = QString::fromStdString(ftype);
        ftTitle[0] = ftTitle[0].toUpper();
        fftTitles_[2]->setText(QString("%1 (r=%2)").arg(ftTitle).arg(cutoff));

        setMetricsText(fftMetrics_, original, result);

        statusBar()->showMessage(
            QString("FFT %1, cutoff=%2")
                .arg(QString::fromStdString(ftype)).arg(cutoff));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "FFT Error", e.what());
    }
}
