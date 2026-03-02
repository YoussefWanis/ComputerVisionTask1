#include "MainWindow.h"
#include "FrequencyDomain.h"
#include "Utils.h"
#include <QFileDialog>
#include <QMessageBox>   // added for QMessageBox::critical

void MainWindow::loadSecondImage() {
    QString path = QFileDialog::getOpenFileName(this, "Select Second Image", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (path.isEmpty()) return;
    secondFull = cv::imread(path.toStdString());
    if (secondFull.empty()) {
        QMessageBox::critical(this, "Error", "Could not load second image.");
        return;
    }
    secondResized = Utils::resizeAspect(secondFull, 512);
    secondValid = true;
    secondFFTValid = false;
    showImage(hybridImg2Label, secondResized);
    updateHybrid();
}

void MainWindow::updateHybrid() {
    if (originalResized.empty() || !secondValid) return;
    if (!fftValid) {
        cv::Mat gray = Utils::toGrayscale(originalResized);
        cachedFFT = FrequencyDomain::computeFFT(gray);
        fftValid = true;
    }
    if (!secondFFTValid) {
        cv::Mat gray = Utils::toGrayscale(secondResized);
        cachedFFTSecond = FrequencyDomain::computeFFT(gray);
        secondFFTValid = true;
    }
    float cutoff1 = hybridCutoff1Slider->value();
    float cutoff2 = hybridCutoff2Slider->value();
    cv::Mat low, high;
    if (hybridModeFirstLow->isChecked()) {
        low = FrequencyDomain::applyLowPass(cachedFFT, cutoff1);
        high = FrequencyDomain::applyHighPass(cachedFFTSecond, cutoff2);
    } else {
        high = FrequencyDomain::applyHighPass(cachedFFT, cutoff1);
        low = FrequencyDomain::applyLowPass(cachedFFTSecond, cutoff2);
    }
    if (low.size() != high.size()) cv::resize(high, high, low.size());
    cv::addWeighted(low, 0.5, high, 0.5, 0, hybridResult);
    showImage(hybridResultLabel, hybridResult);
}

void MainWindow::updateHybridLabels() {
    hybridCutoff1Val->setText(QString::number(hybridCutoff1Slider->value()));
    hybridCutoff2Val->setText(QString::number(hybridCutoff2Slider->value()));
}