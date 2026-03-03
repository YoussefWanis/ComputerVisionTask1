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
 * Handles grayscale images with layout:
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


