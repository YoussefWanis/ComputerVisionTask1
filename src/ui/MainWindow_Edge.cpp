#include "ui/MainWindow.h"
#include <QMessageBox>

// ═══════════════════════════════════════════════════════════════════
//  Canny controls visibility
// ═══════════════════════════════════════════════════════════════════

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

void MainWindow::onApplyEdge() {
    cv::Mat original = requireImage();
    if (original.empty()) return;

    try {
        std::string method = edgeMethod_->currentText().toStdString();

        if (method == "canny") {
            int lo = cannyLo_->value();
            int hi = cannyHi_->value();
            std::string key = "edge_canny_" + std::to_string(lo)
                            + "_" + std::to_string(hi);

            auto& ep = edgeProc_;
            cv::Mat result = model_.getOrCompute(key,
                [&ep, lo, hi](const cv::Mat& img) {
                    return ep.process(img, "canny", "combined", lo, hi);
                });

            // Show: original | canny
            showImageOnLabel(edgeImgs_[0], original);
            edgeTitles_[0]->setText("Original");

            showImageOnLabel(edgeImgs_[1], result);
            edgeTitles_[1]->setText(QString("Canny (L=%1, H=%2)").arg(lo).arg(hi));

            // Hide unused slots
            for (int k = 2; k < 4; ++k) {
                edgeImgs_[k]->setVisible(false);
                edgeTitles_[k]->setVisible(false);
                if (labelSaveButtonMap_.contains(edgeImgs_[k]))
                    labelSaveButtonMap_[edgeImgs_[k]]->setVisible(false);
            }

            setMetricsText(edgeMetrics_, original, result);

        } else {
            // Gradient-based: show original | x | y | combined
            std::string keyX = "edge_" + method + "_x";
            std::string keyY = "edge_" + method + "_y";
            std::string keyC = "edge_" + method + "_combined";

            auto& ep = edgeProc_;
            std::string m = method;

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

            // Show all 4
            for (int k = 0; k < 4; ++k) {
                edgeImgs_[k]->setVisible(true);
                edgeTitles_[k]->setVisible(true);
            }

            showImageOnLabel(edgeImgs_[0], original);
            edgeTitles_[0]->setText("Original");

            QString title = QString::fromStdString(method);
            title[0] = title[0].toUpper();

            showImageOnLabel(edgeImgs_[1], ex);
            edgeTitles_[1]->setText(title + " X");
            showImageOnLabel(edgeImgs_[2], ey);
            edgeTitles_[2]->setText(title + " Y");
            showImageOnLabel(edgeImgs_[3], ec);
            edgeTitles_[3]->setText(title + " Combined");

            setMetricsText(edgeMetrics_, original, ec);
        }

        statusBar()->showMessage("Edge: " + QString::fromStdString(method));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Edge Error", e.what());
    }
}
