/**
 * @file MainWindow_Hybrid.cpp
 * @brief Implements the "Hybrid Images" tab (Tab 5) slot handlers.
 *
 * This file contains two MainWindow slots:
 *   - onLoadImageB()  — loads the second image required for hybrid creation.
 *   - onApplyHybrid() — generates a hybrid image by combining a low-pass
 *                        filtered version of one image with a high-pass
 *                        filtered version of another in the frequency domain.
 *
 * The hybrid-image technique exploits the fact that human perception of
 * spatial frequency content changes with viewing distance: up close you
 * see the high-frequency details, while from far away only the low-
 * frequency content is visible.
 */

#include "ui/MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>

// ═══════════════════════════════════════════════════════════════════
//  Load Image B
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: opens a file dialog to load a second image ("Image B").
 *
 * After loading, Image B is stored in `imageBFull_` and its filename
 * is shown in the info label. If Image A (the primary image) is also
 * loaded, both are displayed side-by-side in the Hybrid tab so the
 * user can see them before creating the hybrid result.
 *
 * Image B is resized to match Image A's dimensions before display,
 * because the FFT-based hybrid algorithm requires both inputs to
 * share the same spatial resolution.
 */
void MainWindow::onLoadImageB() {
    // Open a file dialog filtered to common image formats
    QString path = QFileDialog::getOpenFileName(
        this, "Open Image B", "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All (*)");
    if (path.isEmpty()) return;   // user cancelled the dialog

    // Read the image from disk natively in grayscale
    imageBFull_ = cv::imread(path.toStdString(), cv::IMREAD_GRAYSCALE);
    if (imageBFull_.empty()) {
        QMessageBox::critical(this, "Error", "Could not load Image B.");
        return;
    }

    // Cache the path and update the UI info label with the file name
    imageBPath_ = path;
    lblBInfo_->setText(QFileInfo(path).fileName());
    statusBar()->showMessage("Image B: " + path);

    // If Image A is already loaded, show both images side by side
    cv::Mat original = requireImage();   // returns Image A or empty Mat
    if (!original.empty()) {
        // Resize Image B to the same width and height as Image A
        cv::Mat resizedB;
        cv::resize(imageBFull_, resizedB,
                   cv::Size(original.cols, original.rows));

        // Display Image A and resized Image B; clear the hybrid output slot
        showImageOnLabel(hybridImgs_[0], original);
        hybridTitles_[0]->setText("Image A");
        showImageOnLabel(hybridImgs_[1], resizedB);
        hybridTitles_[1]->setText("Image B");
        showImageOnLabel(hybridImgs_[2], cv::Mat());   // no hybrid yet
        hybridTitles_[2]->clear();
        resetMetrics(hybridMetrics_);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Create Hybrid Image
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: creates a hybrid image from Image A and Image B.
 *
 * The hybrid is computed as:
 *     hybrid = lowpass(lpSource) + ( hpSource − lowpass(hpSource) )
 *
 * Which source acts as LP vs. HP is determined by the `hybridRoles_`
 * combo box (index 0 → A is LP, index 1 → A is HP).  The low-pass
 * and high-pass cutoff radius are read from the `hybridLP_` and
 * `hybridHP_` spin boxes respectively.
 *
 * Processing is delegated to HybridProcessor::create(), which performs
 * the FFT-based filtering internally.  The three display panels are
 * then updated with:
 *   [0] Image A (annotated with its LP/HP role)
 *   [1] Image B (annotated with its LP/HP role)
 *   [2] The resulting hybrid image
 *
 * Quality metrics (e.g. PSNR, MSE) comparing Image A to the hybrid
 * are computed and shown in the metrics label.
 */
void MainWindow::onApplyHybrid() {
    // Ensure Image A is loaded
    cv::Mat original = requireImage();
    if (original.empty()) return;

    // Ensure Image B is loaded
    if (imageBFull_.empty()) {
        QMessageBox::warning(this, "Missing", "Load Image B first.");
        return;
    }

    try {
        // Read user-selected cutoff radii from the spin boxes
        int lp = hybridLP_->value();   // low-pass cutoff radius
        int hp = hybridHP_->value();   // high-pass cutoff radius

        // Resize Image B to match Image A's dimensions
        cv::Mat imgB;
        cv::resize(imageBFull_, imgB,
                   cv::Size(original.cols, original.rows));

        // Determine which image is the low-pass source and which is high-pass
        bool aIsLP = (hybridRoles_->currentIndex() == 0);

        cv::Mat lpImg, hpImg;
        QString lpLabel, hpLabel;

        if (aIsLP) {
            lpImg = original; hpImg = imgB;
            lpLabel = "A";    hpLabel = "B";
        } else {
            lpImg = imgB;     hpImg = original;
            lpLabel = "B";    hpLabel = "A";
        }

        // Delegate the actual frequency-domain hybrid creation
        cv::Mat hybrid = HybridProcessor::create(lpImg, hpImg, lp, hp);

        // ── Update display panels ──────────────────────────────

        // Panel 0: Image A with its role annotation
        showImageOnLabel(hybridImgs_[0], original);
        hybridTitles_[0]->setText(
            QString("Image A (%1)").arg(aIsLP ? "LP" : "HP"));

        // Panel 1: Image B with its role annotation
        showImageOnLabel(hybridImgs_[1], imgB);
        hybridTitles_[1]->setText(
            QString("Image B (%1)").arg(aIsLP ? "HP" : "LP"));

        // Panel 2: resulting hybrid image with cutoff info
        showImageOnLabel(hybridImgs_[2], hybrid);
        hybridTitles_[2]->setText(
            QString("Hybrid (LP[%1]=%2, HP[%3]=%4)")
                .arg(lpLabel).arg(lp).arg(hpLabel).arg(hp));

        // Compute and display quality metrics (PSNR, MSE, etc.)
        setMetricsText(hybridMetrics_, original, hybrid);

        // Update the status bar with a concise summary
        statusBar()->showMessage(
            QString("Hybrid: LP=%1(r=%2), HP=%3(r=%4)")
                .arg(lpLabel).arg(lp).arg(hpLabel).arg(hp));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Hybrid Error", e.what());
    }
}
