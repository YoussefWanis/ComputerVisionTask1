#include "MainWindow.h"
#include "Threshold.h"

void MainWindow::updateGlobalThreshold() {
    if (originalResized.empty()) return;
    int thresh = globalThreshSlider->value();
    globalResult = Threshold::global(originalResized, thresh);
    showImage(globalLabel, globalResult);
}

void MainWindow::updateLocalThreshold() {
    if (originalResized.empty()) return;
    int block = localBlockSlider->value();
    int c = localConstSlider->value();
    localResult = Threshold::local(originalResized, block, c);
    showImage(localLabel, localResult);
}

void MainWindow::updateGlobalLabels() {
    globalThreshVal->setText(QString::number(globalThreshSlider->value()));
}

void MainWindow::updateLocalLabels() {
    localBlockVal->setText(QString::number(localBlockSlider->value()));
    localConstVal->setText(QString::number(localConstSlider->value()));
}