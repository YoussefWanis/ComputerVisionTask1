#include "ui/MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>

// ═══════════════════════════════════════════════════════════════════
//  Load Image B
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onLoadImageB() {
    QString path = QFileDialog::getOpenFileName(
        this, "Open Image B", "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All (*)");
    if (path.isEmpty()) return;

    imageBFull_ = cv::imread(path.toStdString());
    if (imageBFull_.empty()) {
        QMessageBox::critical(this, "Error", "Could not load Image B.");
        return;
    }
    imageBPath_ = path;
    lblBInfo_->setText(QFileInfo(path).fileName());
    statusBar()->showMessage("Image B: " + path);

    // Show both side by side
    cv::Mat original = requireImage();
    if (!original.empty()) {
        // Resize B to match A
        cv::Mat resizedB;
        cv::resize(imageBFull_, resizedB,
                   cv::Size(original.cols, original.rows));

        showImageOnLabel(hybridImgs_[0], original);
        hybridTitles_[0]->setText("Image A");
        showImageOnLabel(hybridImgs_[1], resizedB);
        hybridTitles_[1]->setText("Image B");
        showImageOnLabel(hybridImgs_[2], cv::Mat());
        hybridTitles_[2]->clear();
        resetMetrics(hybridMetrics_);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Create Hybrid Image
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onApplyHybrid() {
    cv::Mat original = requireImage();
    if (original.empty()) return;
    if (imageBFull_.empty()) {
        QMessageBox::warning(this, "Missing", "Load Image B first.");
        return;
    }

    try {
        int lp = hybridLP_->value();
        int hp = hybridHP_->value();

        // Resize B to match A
        cv::Mat imgB;
        cv::resize(imageBFull_, imgB,
                   cv::Size(original.cols, original.rows));

        bool aIsLP = (hybridRoles_->currentIndex() == 0);

        cv::Mat lpImg, hpImg;
        QString lpLabel, hpLabel;

        if (aIsLP) {
            lpImg = original; hpImg = imgB;
            lpLabel = "A"; hpLabel = "B";
        } else {
            lpImg = imgB; hpImg = original;
            lpLabel = "B"; hpLabel = "A";
        }

        cv::Mat hybrid = HybridProcessor::create(lpImg, hpImg, lp, hp);

        // Display
        showImageOnLabel(hybridImgs_[0], original);
        hybridTitles_[0]->setText(
            QString("Image A (%1)").arg(aIsLP ? "LP" : "HP"));

        showImageOnLabel(hybridImgs_[1], imgB);
        hybridTitles_[1]->setText(
            QString("Image B (%1)").arg(aIsLP ? "HP" : "LP"));

        showImageOnLabel(hybridImgs_[2], hybrid);
        hybridTitles_[2]->setText(
            QString("Hybrid (LP[%1]=%2, HP[%3]=%4)")
                .arg(lpLabel).arg(lp).arg(hpLabel).arg(hp));

        setMetricsText(hybridMetrics_, original, hybrid);

        statusBar()->showMessage(
            QString("Hybrid: LP=%1(r=%2), HP=%3(r=%4)")
                .arg(lpLabel).arg(lp).arg(hpLabel).arg(hp));
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Hybrid Error", e.what());
    }
}
