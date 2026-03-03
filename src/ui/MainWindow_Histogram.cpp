/**
 * @file MainWindow_Histogram.cpp
 * @brief Implements Tab 3 (Histograms & Colour) slot handlers for MainWindow.
 *
 * Contains four slots:
 *   - onShowChannels()  — R/G/B channel analysis with per-channel
 *                         histograms and a combined CDF overlay.
 *   - onShowEqualize()  — histogram equalisation with before/after display.
 *   - onShowNormalize() — min-max normalisation with before/after display.
 *   - onShowGrayscale() — grayscale conversion with histogram comparison.
 *
 * Also contains two file-local helper functions:
 *   - hideAllCells()    — hides all cells in the 3×5 histogram grid.
 *   - showCell()        — makes a single grid cell visible.
 *   - drawBeforeAfter() — populates two rows with before/after images,
 *                         per-channel histograms, and CDF overlays.
 */

#include "ui/MainWindow.h"
#include <QMessageBox>

// ═══════════════════════════════════════════════════════════════════
//  Helper — hide all histogram-grid cells
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Hide and clear every cell in the histogram grid.
 *
 * Iterates over all rows × cols cells, clearing the pixmap and text,
 * hiding both the image label and the title label, and hiding the
 * associated Save button.
 *
 * @param self    Pointer to the MainWindow (needed for labelSaveButtonMap_).
 * @param imgs    2-D array of image QLabels.
 * @param titles  2-D array of title QLabels.
 * @param rows    Number of grid rows.
 * @param cols    Number of grid columns.
 */
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

/**
 * @brief Make a single grid cell visible.
 *
 * @param self   MainWindow pointer (unused but kept for consistency).
 * @param img    The image label to show.
 * @param title  The title label to show.
 * @param vis    Visibility flag (default true).
 */
static void showCell(MainWindow* self, QLabel* img, QLabel* title, bool vis = true) {
    img->setVisible(vis);
    title->setVisible(vis);
    // Save button visibility is handled by showImageOnLabel when an image is set
}

// ═══════════════════════════════════════════════════════════════════
//  R / G / B  Channel Analysis
//
//  Grid layout (3 rows × 4 columns):
//  ┌──────────┬────────────┬──────────────┬────────────────┐
//  │ Original │ R channel  │ R histogram  │ Combined CDF   │
//  │   —      │ G channel  │ G histogram  │      —         │
//  │   —      │ B channel  │ B histogram  │      —         │
//  └──────────┴────────────┴──────────────┴────────────────┘
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: display per-channel images, histograms, and CDF overlay.
 *
 * For each of B, G, R:
 *   - Shows a tinted single-channel image (only that channel's colour).
 *   - Renders its 256-bin histogram as a bar chart.
 * Additionally renders a combined B/G/R CDF overlay in the top-right cell.
 *
 * Only works on 3-channel (BGR) images; shows an info dialog for grayscale.
 */
void MainWindow::onShowChannels() {
    cv::Mat original = requireImage();
    if (original.empty()) return;
    if (original.channels() != 3) {
        QMessageBox::information(this, "Info", "Image is already grayscale.");
        return;
    }
    try {
        hideAllCells(this, hcImgs_, hcTitles_, HC_ROWS, HC_COLS);

        // Compute per-channel histograms and CDFs
        auto hists = histProc_.computeChannelHistograms(original);

        // Show the original image in cell (0, 0)
        showCell(this, hcImgs_[0][0], hcTitles_[0][0]);
        showImageOnLabel(hcImgs_[0][0], original);
        hcTitles_[0][0]->setText("Original");

        // Channel names and drawing colours (BGR order matching OpenCV)
        const char* names[] = {"B", "G", "R"};
        cv::Scalar histColors[] = {
            cv::Scalar(255, 0, 0),    // Blue  (in BGR)
            cv::Scalar(0, 255, 0),    // Green
            cv::Scalar(0, 0, 255)     // Red
        };

        // Split the original into B, G, R planes
        std::vector<cv::Mat> channels;
        cv::split(original, channels);

        for (int i = 0; i < 3; ++i) {
            // Create a tinted image: keep only channel i, zero-out others
            cv::Mat tinted;
            std::vector<cv::Mat> planes(3, cv::Mat::zeros(channels[i].size(), CV_8UC1));
            planes[i] = channels[i];
            cv::merge(planes, tinted);

            // Show the tinted channel image in column 1
            showCell(this, hcImgs_[i][1], hcTitles_[i][1]);
            showImageOnLabel(hcImgs_[i][1], tinted);
            hcTitles_[i][1]->setText(QString("%1 Channel").arg(names[i]));

            // Render and show the histogram bar chart in column 2
            cv::Mat histImg = renderHistogramImage(hists[names[i]].histogram,
                                                    histColors[i]);
            showCell(this, hcImgs_[i][2], hcTitles_[i][2]);
            showImageOnLabel(hcImgs_[i][2], histImg);
            hcTitles_[i][2]->setText(QString("%1 Histogram").arg(names[i]));
        }

        // Combined CDF overlay showing all three channels in cell (0, 3)
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
//
//  Grid layout for BGR images (2 rows × 5 columns):
//  ┌──────────┬──────────┬──────────┬──────────┬────────────┐
//  │  Before  │ B hist A │ G hist A │ R hist A │ CDF ovl A  │
//  │  After   │ B hist B │ G hist B │ R hist B │ CDF ovl B  │
//  └──────────┴──────────┴──────────┴──────────┴────────────┘
//
//  Grid layout for grayscale images (2 rows × 3 columns):
//  ┌──────────┬────────────┬──────────┐
//  │  Before  │ Histogram  │   CDF    │
//  │  After   │ Histogram  │   CDF    │
//  └──────────┴────────────┴──────────┘
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Populate the histogram grid with a before/after comparison.
 *
 * Handles both BGR and grayscale images with different layouts:
 * - **BGR**: 2 rows × 5 cols — image, per-channel histograms, CDF overlay.
 * - **Grayscale**: 2 rows × 3 cols — image, histogram, CDF.
 *
 * @param self      MainWindow pointer (needed for display helpers).
 * @param imgs      2-D array of image QLabels (3×5 grid).
 * @param titles    2-D array of title QLabels (3×5 grid).
 * @param before    The "before" (original) image.
 * @param after     The "after" (processed) image.
 * @param titleA    Display title for the before image.
 * @param titleB    Display title for the after image.
 * @param histProc  HistogramProcessor instance for computing histograms.
 */
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
        // ── BGR path: 2 × 5 grid ───────────────────────────────
        auto histsA = histProc.computeChannelHistograms(before);
        auto histsB = histProc.computeChannelHistograms(after);

        // Row 0: before image
        showCell(self, imgs[0][0], titles[0][0]);
        self->showImageOnLabel(imgs[0][0], before);
        titles[0][0]->setText(titleA);

        // Row 1: after image
        showCell(self, imgs[1][0], titles[1][0]);
        self->showImageOnLabel(imgs[1][0], after);
        titles[1][0]->setText(titleB);

        // Collect CDF data for overlay rendering
        std::vector<std::pair<std::vector<double>, cv::Scalar>> cdfA, cdfB;

        for (int i = 0; i < 3; ++i) {
            // Per-channel histograms for before and after
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

        // Combined CDF overlays in column 4
        cv::Mat cA = MainWindow::renderCDFOverlay(cdfA);
        cv::Mat cB = MainWindow::renderCDFOverlay(cdfB);

        showCell(self, imgs[0][4], titles[0][4]);
        self->showImageOnLabel(imgs[0][4], cA);
        titles[0][4]->setText(titleA + " CDF");

        showCell(self, imgs[1][4], titles[1][4]);
        self->showImageOnLabel(imgs[1][4], cB);
        titles[1][4]->setText(titleB + " CDF");

    } else {
        // ── Grayscale path: 2 × 3 grid ─────────────────────────
        // Compute histogram + CDF for grayscale images using Processor
        ChannelHistData dataA = HistogramProcessor::computeHistogramAndCDF(before);
        ChannelHistData dataB = HistogramProcessor::computeHistogramAndCDF(after);
        
        auto hA = dataA.histogram;
        auto cA = dataA.cdf;
        auto hB = dataB.histogram;
        auto cB = dataB.cdf;

        // Row 0: before — image, histogram, CDF
        showCell(self, imgs[0][0], titles[0][0]);
        self->showImageOnLabel(imgs[0][0], before);
        titles[0][0]->setText(titleA);

        showCell(self, imgs[0][1], titles[0][1]);
        self->showImageOnLabel(imgs[0][1], MainWindow::renderHistogramImage(hA, cv::Scalar(128, 128, 128)));
        titles[0][1]->setText(titleA + " Histogram");

        showCell(self, imgs[0][2], titles[0][2]);
        self->showImageOnLabel(imgs[0][2], MainWindow::renderCDFImage(cA, cv::Scalar(128, 128, 128)));
        titles[0][2]->setText(titleA + " CDF");

        // Row 1: after — image, histogram, CDF
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

/**
 * @brief Slot: equalise the image histogram and display before/after.
 *
 * Uses HistogramProcessor::equalize() (cached) and delegates the
 * grid population to drawBeforeAfter().
 */
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

/**
 * @brief Slot: normalise the image (min-max) and display before/after.
 *
 * Uses HistogramProcessor::normalize() (cached) and delegates the
 * grid population to drawBeforeAfter().
 */
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
//  Grid layout (2 rows × 5 columns):
//  ┌────────────────┬───────────┬───────────┬───────────┬───────────┐
//  │ Original RGB   │ B hist    │ G hist    │ R hist    │ B/G/R CDF │
//  │ Grayscale      │ Gray hist │ Gray CDF  │    —      │     —     │
//  └────────────────┴───────────┴───────────┴───────────┴───────────┘
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: convert to grayscale and compare histograms.
 *
 * Row 0 shows the original RGB image with its per-channel histograms
 * and a combined B/G/R CDF overlay.
 *
 * Row 1 shows the grayscale conversion with its single-channel
 * histogram and CDF.
 *
 * Only works on 3-channel (BGR) images; shows an info dialog for grayscale.
 */
void MainWindow::onShowGrayscale() {
    cv::Mat original = requireImage();
    if (original.empty()) return;
    if (original.channels() != 3) {
        QMessageBox::information(this, "Info", "Image is already grayscale.");
        return;
    }
    try {
        // Convert to grayscale (cached)
        cv::Mat gray = model_.getOrCompute("grayscale",
            [](const cv::Mat& img) {
                return ColorProcessor::toGrayscale(img);
            });

        hideAllCells(this, hcImgs_, hcTitles_, HC_ROWS, HC_COLS);

        // Compute per-channel histograms for the original image
        auto hists = histProc_.computeChannelHistograms(original);

        const char* names[] = {"B", "G", "R"};
        cv::Scalar colors[] = {
            cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)
        };

        // ── Row 0: Original + per-channel histograms + CDF ──────
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

        // ── Row 1: Grayscale + gray histogram + gray CDF ────────
        showCell(this, hcImgs_[1][0], hcTitles_[1][0]);
        showImageOnLabel(hcImgs_[1][0], gray);
        hcTitles_[1][0]->setText("Grayscale (BT.601)");

        // Compute grayscale histogram and CDF using Processor
        ChannelHistData grayData = HistogramProcessor::computeHistogramAndCDF(gray);
        std::vector<int>& grayHist = grayData.histogram;
        std::vector<double>& grayCdf = grayData.cdf;

        // Render and display the grayscale histogram
        cv::Mat gh = renderHistogramImage(grayHist, cv::Scalar(128, 128, 128));
        showCell(this, hcImgs_[1][1], hcTitles_[1][1]);
        showImageOnLabel(hcImgs_[1][1], gh);
        hcTitles_[1][1]->setText("Gray Histogram");

        // Render and display the grayscale CDF
        cv::Mat gc = renderCDFImage(grayCdf, cv::Scalar(128, 128, 128));
        showCell(this, hcImgs_[1][2], hcTitles_[1][2]);
        showImageOnLabel(hcImgs_[1][2], gc);
        hcTitles_[1][2]->setText("Gray CDF");

        statusBar()->showMessage("Grayscale conversion done");
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", e.what());
    }
}
