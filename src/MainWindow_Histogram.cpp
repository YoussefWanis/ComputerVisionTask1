#include "MainWindow.h"
#include "Histogram.h"
#include "Utils.h"
#include <QMessageBox>

void MainWindow::updateRGBChannels() {
    if (originalResized.empty() || originalResized.channels() != 3) {
        QMessageBox::warning(this, "Warning", "RGB channels require a color image.");
        return;
    }
    std::vector<cv::Mat> bgr;
    cv::split(originalResized, bgr);

    // Convert single channels to BGR for display (as colored images)
    cv::cvtColor(bgr[2], redChannelImage, cv::COLOR_GRAY2BGR);
    cv::cvtColor(bgr[1], greenChannelImage, cv::COLOR_GRAY2BGR);
    cv::cvtColor(bgr[0], blueChannelImage, cv::COLOR_GRAY2BGR);

    showImage(redImageLabel, redChannelImage);
    showImage(greenImageLabel, greenChannelImage);
    showImage(blueImageLabel, blueChannelImage);

    // Compute and draw histograms
    redHistImage = Utils::drawHistogram(Utils::computeHist(bgr[2]), cv::Scalar(0,0,255), 150, 150);
    greenHistImage = Utils::drawHistogram(Utils::computeHist(bgr[1]), cv::Scalar(0,255,0), 150, 150);
    blueHistImage = Utils::drawHistogram(Utils::computeHist(bgr[0]), cv::Scalar(255,0,0), 150, 150);

    showImage(redHistLabel, redHistImage);
    showImage(greenHistLabel, greenHistImage);
    showImage(blueHistLabel, blueHistImage);
}

void MainWindow::updateEqualize() {
    if (originalResized.empty()) return;

    equalizedResult = Histogram::equalize(originalResized);
    showImage(eqImageLabel, equalizedResult);

    cv::Mat hist = Utils::computeHist(Utils::toGrayscale(equalizedResult));
    cv::Mat cdf = Utils::computeCumulativeHist(hist);

    eqHistImage = Utils::drawHistogram(hist, cv::Scalar(0,0,0), 150, 150);
    eqCDFImage = Utils::drawHistogram(cdf, cv::Scalar(0,0,0), 150, 150);

    showImage(eqHistLabel, eqHistImage);
    showImage(eqCDFLabel, eqCDFImage);
}

void MainWindow::updateNormalize() {
    if (originalResized.empty()) return;

    normalizedResult = Histogram::normalize(originalResized);
    showImage(normImageLabel, normalizedResult);

    cv::Mat hist = Utils::computeHist(Utils::toGrayscale(normalizedResult));
    cv::Mat cdf = Utils::computeCumulativeHist(hist);

    normHistImage = Utils::drawHistogram(hist, cv::Scalar(0,0,0), 150, 150);
    normCDFImage = Utils::drawHistogram(cdf, cv::Scalar(0,0,0), 150, 150);

    showImage(normHistLabel, normHistImage);
    showImage(normCDFLabel, normCDFImage);
}