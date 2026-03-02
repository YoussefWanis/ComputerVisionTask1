#include "ui/MainWindow.h"
#include <QMessageBox>

// ═══════════════════════════════════════════════════════════════════
//  Helper — hide all histogram-grid cells
// ═══════════════════════════════════════════════════════════════════

static void hideAllCells(MainWindow* self, QLabel* imgs[][5], QLabel* titles[][5],
                         int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            imgs[r][c]->clear();
            imgs[r][c]->setVisible(false);
            titles[r][c]->clear();
            titles[r][c]->setVisible(false);
            if (self->labelSaveButtonMap_.contains(imgs[r][c]))
                self->labelSaveButtonMap_[imgs[r][c]]->setVisible(false);
        }
}

static void showCell(MainWindow* self, QLabel* img, QLabel* title, bool vis = true) {
    img->setVisible(vis);
    title->setVisible(vis);
    // Save button visibility is handled by showImageOnLabel when image is set
}

// ═══════════════════════════════════════════════════════════════════
//  R / G / B  Channel Analysis
//
//  Layout (3 × 4):
//  [Original]  [R channel]  [R histogram]  [Combined CDF]
//  [  —  ]     [G channel]  [G histogram]  [  —  ]
//  [  —  ]     [B channel]  [B histogram]  [  —  ]
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onShowChannels() {
    cv::Mat original = requireImage();
    if (original.empty()) return;
    if (original.channels() != 3) {
        QMessageBox::information(this, "Info", "Image is already grayscale.");
        return;
    }
    try {
        hideAllCells(this, hcImgs_, hcTitles_, HC_ROWS, HC_COLS);

        auto hists = histProc_.computeChannelHistograms(original);

        // Original in (0,0)
        showCell(this, hcImgs_[0][0], hcTitles_[0][0]);
        showImageOnLabel(hcImgs_[0][0], original);
        hcTitles_[0][0]->setText("Original");

        // Channel images (tinted) and histograms
        const char* names[] = {"B", "G", "R"};
        // BGR colours for drawing histograms and tinting
        cv::Scalar histColors[] = {
            cv::Scalar(255, 0, 0),    // Blue  (BGR)
            cv::Scalar(0, 255, 0),    // Green
            cv::Scalar(0, 0, 255)     // Red
        };

        std::vector<cv::Mat> channels;
        cv::split(original, channels);  // B, G, R

        for (int i = 0; i < 3; ++i) {
            // Channel image (tinted)
            cv::Mat tinted;
            std::vector<cv::Mat> planes(3, cv::Mat::zeros(channels[i].size(), CV_8UC1));
            planes[i] = channels[i];
            cv::merge(planes, tinted);

            showCell(this, hcImgs_[i][1], hcTitles_[i][1]);
            showImageOnLabel(hcImgs_[i][1], tinted);
            hcTitles_[i][1]->setText(QString("%1 Channel").arg(names[i]));

            // Histogram
            cv::Mat histImg = renderHistogramImage(hists[names[i]].histogram,
                                                    histColors[i]);
            showCell(this, hcImgs_[i][2], hcTitles_[i][2]);
            showImageOnLabel(hcImgs_[i][2], histImg);
            hcTitles_[i][2]->setText(QString("%1 Histogram").arg(names[i]));
        }

        // Combined CDF overlay in (0,3)
        std::vector<std::pair<std::vector<double>, cv::Scalar>> cdfData;
        for (int i = 0; i < 3; ++i)
            cdfData.push_back({hists[names[i]].cdf, histColors[i]});

        cv::Mat cdfImg = renderCDFOverlay(cdfData);
        showCell(this, hcImgs_[0][3], hcTitles_[0][3]);
        showImageOnLabel(hcImgs_[0][3], cdfImg);
        hcTitles_[0][3]->setText("B/G/R CDF");

        statusBar()->showMessage("Channel analysis displayed");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", e.what());
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Helper — Before/After with per-channel histograms + CDF
//  Layout (2 × 5):
//  [Before]  [B hist A]  [G hist A]  [R hist A]  [CDF overlay A]
//  [After ]  [B hist B]  [G hist B]  [R hist B]  [CDF overlay B]
// ═══════════════════════════════════════════════════════════════════

static void drawBeforeAfter(
        MainWindow* self,
        QLabel* imgs[][5], QLabel* titles[][5],
        const cv::Mat& before, const cv::Mat& after,
        const QString& titleA, const QString& titleB,
        HistogramProcessor& histProc) {
    hideAllCells(self, imgs, titles, 3, 5);

    const char* names[] = {"B", "G", "R"};
    cv::Scalar colors[] = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)
    };

    if (before.channels() == 3) {
        auto histsA = histProc.computeChannelHistograms(before);
        auto histsB = histProc.computeChannelHistograms(after);

        // Row 0: before
        showCell(self, imgs[0][0], titles[0][0]);
        self->showImageOnLabel(imgs[0][0], before);
        titles[0][0]->setText(titleA);

        // Row 1: after
        showCell(self, imgs[1][0], titles[1][0]);
        self->showImageOnLabel(imgs[1][0], after);
        titles[1][0]->setText(titleB);

        std::vector<std::pair<std::vector<double>, cv::Scalar>> cdfA, cdfB;

        for (int i = 0; i < 3; ++i) {
            // Histograms
            cv::Mat hA = MainWindow::renderHistogramImage(histsA[names[i]].histogram, colors[i]);
            cv::Mat hB = MainWindow::renderHistogramImage(histsB[names[i]].histogram, colors[i]);

            showCell(self, imgs[0][1 + i], titles[0][1 + i]);
            self->showImageOnLabel(imgs[0][1 + i], hA);
            titles[0][1 + i]->setText(QString("%1 %2").arg(titleA, names[i]));

            showCell(self, imgs[1][1 + i], titles[1][1 + i]);
            self->showImageOnLabel(imgs[1][1 + i], hB);
            titles[1][1 + i]->setText(QString("%1 %2").arg(titleB, names[i]));

            cdfA.push_back({histsA[names[i]].cdf, colors[i]});
            cdfB.push_back({histsB[names[i]].cdf, colors[i]});
        }

        // CDF overlays
        cv::Mat cA = MainWindow::renderCDFOverlay(cdfA);
        cv::Mat cB = MainWindow::renderCDFOverlay(cdfB);

        showCell(self, imgs[0][4], titles[0][4]);
        self->showImageOnLabel(imgs[0][4], cA);
        titles[0][4]->setText(titleA + " CDF");

        showCell(self, imgs[1][4], titles[1][4]);
        self->showImageOnLabel(imgs[1][4], cB);
        titles[1][4]->setText(titleB + " CDF");

    } else {
        // Grayscale: 2 × 3  [Image | Histogram | CDF]
        auto makeGrayHist = [](const cv::Mat& img)
                -> std::pair<std::vector<int>, std::vector<double>> {
            std::vector<int> hist(256, 0);
            for (int i = 0; i < img.rows; ++i) {
                const uchar* row = img.ptr<uchar>(i);
                for (int j = 0; j < img.cols; ++j)
                    hist[row[j]]++;
            }
            std::vector<double> cdf(256);
            double cumSum = 0, total = img.rows * img.cols;
            for (int k = 0; k < 256; ++k) {
                cumSum += hist[k];
                cdf[k] = cumSum / total;
            }
            return {hist, cdf};
        };

        auto [hA, cA] = makeGrayHist(before);
        auto [hB, cB] = makeGrayHist(after);

        showCell(self, imgs[0][0], titles[0][0]);
        self->showImageOnLabel(imgs[0][0], before);
        titles[0][0]->setText(titleA);

        showCell(self, imgs[0][1], titles[0][1]);
        self->showImageOnLabel(imgs[0][1], MainWindow::renderHistogramImage(hA, cv::Scalar(128, 128, 128)));
        titles[0][1]->setText(titleA + " Histogram");

        showCell(self, imgs[0][2], titles[0][2]);
        self->showImageOnLabel(imgs[0][2], MainWindow::renderCDFImage(cA, cv::Scalar(128, 128, 128)));
        titles[0][2]->setText(titleA + " CDF");

        showCell(self, imgs[1][0], titles[1][0]);
        self->showImageOnLabel(imgs[1][0], after);
        titles[1][0]->setText(titleB);

        showCell(self, imgs[1][1], titles[1][1]);
        self->showImageOnLabel(imgs[1][1], MainWindow::renderHistogramImage(hB, cv::Scalar(128, 128, 128)));
        titles[1][1]->setText(titleB + " Histogram");

        showCell(self, imgs[1][2], titles[1][2]);
        self->showImageOnLabel(imgs[1][2], MainWindow::renderCDFImage(cB, cv::Scalar(128, 128, 128)));
        titles[1][2]->setText(titleB + " CDF");
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Equalize
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onShowEqualize() {
    cv::Mat original = requireImage();
    if (original.empty()) return;
    try {
        auto& hp = histProc_;
        cv::Mat equalized = model_.getOrCompute("equalized",
            [&hp](const cv::Mat& img) { return hp.equalize(img); });

        drawBeforeAfter(this, hcImgs_, hcTitles_,
                        original, equalized,
                        "Original", "Equalized", histProc_);
        statusBar()->showMessage("Histogram equalization applied");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", e.what());
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Normalize
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onShowNormalize() {
    cv::Mat original = requireImage();
    if (original.empty()) return;
    try {
        cv::Mat normalized = model_.getOrCompute("normalized",
            [](const cv::Mat& img) {
                return HistogramProcessor::normalize(img);
            });

        drawBeforeAfter(this, hcImgs_, hcTitles_,
                        original, normalized,
                        "Original", "Normalized", histProc_);
        statusBar()->showMessage("Normalization applied");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", e.what());
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Grayscale
//
//  Layout (2 × 5):
//  [Original RGB]  [B hist]  [G hist]  [R hist]  [B/G/R CDF]
//  [Grayscale   ]  [Gray hist]  [Gray CDF]  [ — ]  [ — ]
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onShowGrayscale() {
    cv::Mat original = requireImage();
    if (original.empty()) return;
    if (original.channels() != 3) {
        QMessageBox::information(this, "Info", "Image is already grayscale.");
        return;
    }
    try {
        cv::Mat gray = model_.getOrCompute("grayscale",
            [](const cv::Mat& img) {
                return ColorProcessor::toGrayscale(img);
            });

        hideAllCells(this, hcImgs_, hcTitles_, HC_ROWS, HC_COLS);

        auto hists = histProc_.computeChannelHistograms(original);

        const char* names[] = {"B", "G", "R"};
        cv::Scalar colors[] = {
            cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)
        };

        // Row 0: Original + per-channel histograms + combined CDF
        showCell(this, hcImgs_[0][0], hcTitles_[0][0]);
        showImageOnLabel(hcImgs_[0][0], original);
        hcTitles_[0][0]->setText("Original RGB");

        std::vector<std::pair<std::vector<double>, cv::Scalar>> cdfData;
        for (int i = 0; i < 3; ++i) {
            cv::Mat h = renderHistogramImage(hists[names[i]].histogram, colors[i]);
            showCell(this, hcImgs_[0][1 + i], hcTitles_[0][1 + i]);
            showImageOnLabel(hcImgs_[0][1 + i], h);
            hcTitles_[0][1 + i]->setText(QString("%1 Histogram").arg(names[i]));
            cdfData.push_back({hists[names[i]].cdf, colors[i]});
        }

        cv::Mat cdf = renderCDFOverlay(cdfData);
        showCell(this, hcImgs_[0][4], hcTitles_[0][4]);
        showImageOnLabel(hcImgs_[0][4], cdf);
        hcTitles_[0][4]->setText("B/G/R CDF");

        // Row 1: Grayscale + gray histogram + gray CDF
        showCell(this, hcImgs_[1][0], hcTitles_[1][0]);
        showImageOnLabel(hcImgs_[1][0], gray);
        hcTitles_[1][0]->setText("Grayscale (BT.601)");

        // Gray histogram
        std::vector<int> grayHist(256, 0);
        for (int i = 0; i < gray.rows; ++i) {
            const uchar* row = gray.ptr<uchar>(i);
            for (int j = 0; j < gray.cols; ++j)
                grayHist[row[j]]++;
        }
        std::vector<double> grayCdf(256);
        double cumSum = 0, total = gray.rows * gray.cols;
        for (int k = 0; k < 256; ++k) {
            cumSum += grayHist[k];
            grayCdf[k] = cumSum / total;
        }

        cv::Mat gh = renderHistogramImage(grayHist, cv::Scalar(128, 128, 128));
        showCell(this, hcImgs_[1][1], hcTitles_[1][1]);
        showImageOnLabel(hcImgs_[1][1], gh);
        hcTitles_[1][1]->setText("Gray Histogram");

        cv::Mat gc = renderCDFImage(grayCdf, cv::Scalar(128, 128, 128));
        showCell(this, hcImgs_[1][2], hcTitles_[1][2]);
        showImageOnLabel(hcImgs_[1][2], gc);
        hcTitles_[1][2]->setText("Gray CDF");

        statusBar()->showMessage("Grayscale conversion done");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", e.what());
    }
}
