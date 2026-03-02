#include "MainWindow.h"
#include "FrequencyDomain.h"
#include "Utils.h"

void MainWindow::updateLowPass() {
    if (originalResized.empty()) return;
    if (!fftValid) {
        cv::Mat gray = Utils::toGrayscale(originalResized);
        cachedFFT = FrequencyDomain::computeFFT(gray);
        fftValid = true;
    }
    float cutoff = lowPassCutoffSlider->value();
    lowPassResult = FrequencyDomain::applyLowPass(cachedFFT, cutoff);
    showImage(lowPassLabel, lowPassResult);
}

void MainWindow::updateHighPass() {
    if (originalResized.empty()) return;
    if (!fftValid) {
        cv::Mat gray = Utils::toGrayscale(originalResized);
        cachedFFT = FrequencyDomain::computeFFT(gray);
        fftValid = true;
    }
    float cutoff = highPassCutoffSlider->value();
    highPassResult = FrequencyDomain::applyHighPass(cachedFFT, cutoff);
    showImage(highPassLabel, highPassResult);
}

void MainWindow::updateLowPassLabels() {
    lowPassCutoffVal->setText(QString::number(lowPassCutoffSlider->value()));
}

void MainWindow::updateHighPassLabels() {
    highPassCutoffVal->setText(QString::number(highPassCutoffSlider->value()));
}