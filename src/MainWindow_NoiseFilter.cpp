#include "MainWindow.h"
#include "Noise.h"
#include "Filters.h"

void MainWindow::onNoiseTypeChanged(int index) {
    nfUniformWidget->setVisible(index == 0);
    nfGaussianWidget->setVisible(index == 1);
    nfSPWidget->setVisible(index == 2);
    updateNoiseFilter();
}

void MainWindow::onFilterTypeChanged(int index) {
    nfAvgWidget->setVisible(index == 0);
    nfGaussFilterWidget->setVisible(index == 1);
    nfMedianWidget->setVisible(index == 2);
    updateNoiseFilter();
}

void MainWindow::updateNoiseFilter() {
    if (originalResized.empty()) return;

    int noiseIdx = nfNoiseCombo->currentIndex();
    switch (noiseIdx) {
    case 0:
        nfNoisyImage = Noise::addUniform(originalResized, nfUniformLow->value(), nfUniformHigh->value());
        break;
    case 1:
        nfNoisyImage = Noise::addGaussian(originalResized, nfGaussianMean->value(), nfGaussianStd->value());
        break;
    case 2:
        nfNoisyImage = Noise::addSaltPepper(originalResized, nfSPProb->value() / 100.0);
        break;
    }
    showImage(nfNoisyLabel, nfNoisyImage);

    int filterIdx = nfFilterCombo->currentIndex();
    switch (filterIdx) {
    case 0: {
        int k = nfAvgKernel->value();
        if (k % 2 == 0) k++;
        nfFilteredImage = Filters::average(nfNoisyImage, k);
        break;
    }
    case 1: {
        int k = nfGaussKernel->value();
        if (k % 2 == 0) k++;
        double sigma = nfGaussSigma->value() / 10.0;
        nfFilteredImage = Filters::gaussian(nfNoisyImage, k, sigma);
        break;
    }
    case 2: {
        int k = nfMedianKernel->value();
        if (k % 2 == 0) k++;
        nfFilteredImage = Filters::median(nfNoisyImage, k);
        break;
    }
    }
    showImage(nfFilteredLabel, nfFilteredImage);

    double mse, psnr, snr;
    computeMetrics(originalResized, nfFilteredImage, mse, psnr, snr);
    nfMetricsLabel->setText(QString("MSE: %1\nPSNR: %2 dB\nSNR: %3 dB")
                             .arg(mse, 0, 'f', 2)
                             .arg(psnr, 0, 'f', 2)
                             .arg(snr, 0, 'f', 2));
}

void MainWindow::updateNoiseFilterLabels() {
    nfUniformLowVal->setText(QString::number(nfUniformLow->value()));
    nfUniformHighVal->setText(QString::number(nfUniformHigh->value()));
    nfGaussianMeanVal->setText(QString::number(nfGaussianMean->value()));
    nfGaussianStdVal->setText(QString::number(nfGaussianStd->value()));
    nfSPProbVal->setText(QString::number(nfSPProb->value() / 100.0, 'f', 2));
    nfAvgKernelVal->setText(QString::number(nfAvgKernel->value()));
    nfGaussKernelVal->setText(QString::number(nfGaussKernel->value()));
    nfGaussSigmaVal->setText(QString::number(nfGaussSigma->value() / 10.0, 'f', 1));
    nfMedianKernelVal->setText(QString::number(nfMedianKernel->value()));
}