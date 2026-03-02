#include "MainWindow.h"
#include "Noise.h"

// --- Noise tab: processing (slider release) ---
void MainWindow::updateUniformNoise() {
    if (originalResized.empty()) return;
    double low = uniformLowSlider->value();
    double high = uniformHighSlider->value();
    uniformResult = Noise::addUniform(originalResized, low, high);
    showImage(uniformLabel, uniformResult);
}

void MainWindow::updateGaussianNoise() {
    if (originalResized.empty()) return;
    double mean = gaussianMeanSlider->value();
    double std = gaussianStdSlider->value();
    gaussianResult = Noise::addGaussian(originalResized, mean, std);
    showImage(gaussianLabel, gaussianResult);
}

void MainWindow::updateSaltPepperNoise() {
    if (originalResized.empty()) return;
    double prob = spProbSlider->value() / 100.0;
    spResult = Noise::addSaltPepper(originalResized, prob);
    showImage(spLabel, spResult);
}

// --- Noise tab: label updates (slider value changed) ---
void MainWindow::updateUniformLabels() {
    uniformLowVal->setText(QString::number(uniformLowSlider->value()));
    uniformHighVal->setText(QString::number(uniformHighSlider->value()));
}

void MainWindow::updateGaussianLabels() {
    gaussianMeanVal->setText(QString::number(gaussianMeanSlider->value()));
    gaussianStdVal->setText(QString::number(gaussianStdSlider->value()));
}

void MainWindow::updateSPLabels() {
    double prob = spProbSlider->value() / 100.0;
    spProbVal->setText(QString::number(prob, 'f', 2));
}