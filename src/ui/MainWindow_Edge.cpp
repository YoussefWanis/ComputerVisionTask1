/**
 * @file MainWindow_Edge.cpp
 * @brief Implements Tab 2 (Edge Detection) slot handlers for MainWindow.
 *
 * Contains two slots:
 *   - onEdgeMethodChanged() — toggles Canny threshold controls visibility.
 *   - onApplyEdge()         — runs the selected edge detector and
 *                             displays the results (up to 4 panels).
 *
 * For gradient-based methods (Sobel, Roberts, Prewitt), four panels are
 * shown: Original, X gradient, Y gradient, and Combined magnitude.
 * For Canny, only two panels are shown: Original and the binary edge map.
 */

#include "ui/MainWindow.h"
#include <QMessageBox>

// ═══════════════════════════════════════════════════════════════════
//  Canny controls visibility
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: show or hide the Canny low/high threshold controls.
 *
 * Called whenever the edge-method combo box selection changes.
 * The Canny thresholds are only relevant when "canny" is selected;
 * they are hidden for gradient-based methods.
 *
 * @param method  The newly selected method string (e.g. "sobel", "canny").
 */
void MainWindow::onEdgeMethodChanged(const QString& method) {
    bool vis = (method == "canny");
    lblCannyLo_->setVisible(vis);
    cannyLo_->setVisible(vis);
    lblCannyHi_->setVisible(vis);
    cannyHi_->setVisible(vis);
}

// ═══════════════════════════════════════════════════════════════════
//  Apply Edge Detection
// ═══════════════════════════════════════════════════════════════════

/**
 * @brief Slot: run edge detection and display the results.
 *
 * Behaviour depends on the selected method:
 *
 * **Canny:**
 *   - Computes a binary edge map using the low/high thresholds from
 *     the spin boxes.
 *   - Displays 2 panels: [Original | Canny edges].
 *   - Hides panels 3 and 4.
 *
 * **Gradient-based (Sobel / Roberts / Prewitt):**
 *   - Computes three results: X gradient, Y gradient, combined magnitude.
 *   - Uses the ImageModel cache (getOrCompute) so repeated requests
 *     with the same parameters are served instantly.
 *   - Displays 4 panels: [Original | X | Y | Combined].
 *
 * Quality metrics (MSE, PSNR, SNR) are computed between the original
 * image and the primary edge output (Canny result or Combined).
 */
void MainWindow::onApplyEdge() {
    cv::Mat original = requireImage();
    if (original.empty()) return;

    try {
        std::string method = edgeMethod_->currentText().toStdString();

        if (method == "canny") {
            // ── Canny path ──────────────────────────────────────
            int lo = cannyLo_->value();
            int hi = cannyHi_->value();

            // Cache key encodes the method and both thresholds
            std::string key = "edge_canny_" + std::to_string(lo)
                            + "_" + std::to_string(hi);

            auto& ep = edgeProc_;
            cv::Mat result = model_.getOrCompute(key,
                [&ep, lo, hi](const cv::Mat& img) {
                    return ep.process(img, "canny", "combined", lo, hi);
                });

            // Display: Original | Canny result
            showImageOnLabel(edgeImgs_[0], original);
            edgeTitles_[0]->setText("Original");

            showImageOnLabel(edgeImgs_[1], result);
            edgeTitles_[1]->setText(QString("Canny (L=%1, H=%2)").arg(lo).arg(hi));

            // Hide unused panels (3 and 4)
            for (int k = 2; k < 4; ++k) {
                edgeImgs_[k]->setVisible(false);
                edgeTitles_[k]->setVisible(false);
                if (labelSaveButtonMap_.contains(edgeImgs_[k]))
                    labelSaveButtonMap_[edgeImgs_[k]]->setVisible(false);
            }

            // Quality metrics: original vs. Canny edges
            setMetricsText(edgeMetrics_, original, result);

        } else {
            // ── Gradient-based path (Sobel / Roberts / Prewitt) ─
            // Build cache keys for X, Y, and Combined directions
            std::string keyX = "edge_" + method + "_x";
            std::string keyY = "edge_" + method + "_y";
            std::string keyC = "edge_" + method + "_combined";

            auto& ep = edgeProc_;
            std::string m = method;

            // Compute (or retrieve from cache) all three directions
            cv::Mat ex = model_.getOrCompute(keyX,
                [&ep, m](const cv::Mat& img) {
                    return ep.process(img, m, "x");
                });
            cv::Mat ey = model_.getOrCompute(keyY,
                [&ep, m](const cv::Mat& img) {
                    return ep.process(img, m, "y");
                });
            cv::Mat ec = model_.getOrCompute(keyC,
                [&ep, m](const cv::Mat& img) {
                    return ep.process(img, m, "combined");
                });

            // Ensure all 4 panels are visible
            for (int k = 0; k < 4; ++k) {
                edgeImgs_[k]->setVisible(true);
                edgeTitles_[k]->setVisible(true);
            }

            // Display: Original | X gradient | Y gradient | Combined
            showImageOnLabel(edgeImgs_[0], original);
            edgeTitles_[0]->setText("Original");

            // Capitalise the method name for display titles
            QString title = QString::fromStdString(method);
            title[0] = title[0].toUpper();

            showImageOnLabel(edgeImgs_[1], ex);
            edgeTitles_[1]->setText(title + " X");
            showImageOnLabel(edgeImgs_[2], ey);
            edgeTitles_[2]->setText(title + " Y");
            showImageOnLabel(edgeImgs_[3], ec);
            edgeTitles_[3]->setText(title + " Combined");

            // Quality metrics: original vs. combined edge magnitude
            setMetricsText(edgeMetrics_, original, ec);
        }

        statusBar()->showMessage("Edge: " + QString::fromStdString(method));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Edge Error", e.what());
    }
}
