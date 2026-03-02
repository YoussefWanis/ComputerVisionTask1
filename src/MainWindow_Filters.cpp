#include "MainWindow.h"
#include "Filters.h"

void MainWindow::updateAverageFilter() {
    if (originalResized.empty()) return;
    int k = avgKernelSlider->value();
    if (k % 2 == 0) k++;
    averageResult = Filters::average(originalResized, k);
    showImage(averageLabel, averageResult);
}

void MainWindow::updateGaussianFilter() {
    if (originalResized.empty()) return;
    int k = gaussKernelSlider->value();
    if (k % 2 == 0) k++;
    double sigma = gaussSigmaSlider->value() / 10.0;
    gaussFilterResult = Filters::gaussian(originalResized, k, sigma);
    showImage(gaussFilterLabel, gaussFilterResult);
}

void MainWindow::updateMedianFilter() {
    if (originalResized.empty()) return;
    int k = medianKernelSlider->value();
    if (k % 2 == 0) k++;
    medianResult = Filters::median(originalResized, k);
    showImage(medianLabel, medianResult);
}

void MainWindow::updateAvgLabels() {
    avgKernelVal->setText(QString::number(avgKernelSlider->value()));
}

void MainWindow::updateGaussFilterLabels() {
    gaussKernelVal->setText(QString::number(gaussKernelSlider->value()));
    double sigma = gaussSigmaSlider->value() / 10.0;
    gaussSigmaVal->setText(QString::number(sigma, 'f', 1));
}

void MainWindow::updateMedianLabels() {
    medianKernelVal->setText(QString::number(medianKernelSlider->value()));
}