#include "MainWindow.h"
#include "EdgeDetection.h"

void MainWindow::updateSobel() {
    if (originalResized.empty()) return;
    sobelResult = EdgeDetection::sobel(originalResized);
    showImage(sobelLabel, sobelResult);
}

void MainWindow::updateSobelX() {
    if (originalResized.empty()) return;
    sobelXResult = EdgeDetection::sobelX(originalResized);
    showImage(sobelXLabel, sobelXResult);
}

void MainWindow::updateSobelY() {
    if (originalResized.empty()) return;
    sobelYResult = EdgeDetection::sobelY(originalResized);
    showImage(sobelYLabel, sobelYResult);
}

void MainWindow::updateRoberts() {
    if (originalResized.empty()) return;
    robertsResult = EdgeDetection::roberts(originalResized);
    showImage(robertsLabel, robertsResult);
}

void MainWindow::updatePrewitt() {
    if (originalResized.empty()) return;
    prewittResult = EdgeDetection::prewitt(originalResized);
    showImage(prewittLabel, prewittResult);
}

void MainWindow::updateCanny() {
    if (originalResized.empty()) return;
    int low = cannyLowSlider->value();
    int high = cannyHighSlider->value();
    cannyResult = EdgeDetection::canny(originalResized, low, high);
    showImage(cannyLabel, cannyResult);
}

void MainWindow::updateCannyLabels() {
    cannyLowVal->setText(QString::number(cannyLowSlider->value()));
    cannyHighVal->setText(QString::number(cannyHighSlider->value()));
}