#include "ui/MainWindow.h"
#include "ui/ZoomableImageDialog.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QSizePolicy>
#include <QWidget>
#include <QImage>
#include <QPixmap>
#include <QAction>
#include <QMenu>
#include <QScrollArea>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════
//  Constructor
// ═══════════════════════════════════════════════════════════════════

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , noiseProc_(0)
{
    setWindowTitle("CV Task 1 — Image Processing Pipeline");
    setMinimumSize(1200, 750);
    buildUI();
}

// ═══════════════════════════════════════════════════════════════════
//  UI construction
// ═══════════════════════════════════════════════════════════════════

void MainWindow::buildUI() {
    auto* central = new QWidget(this);
    setCentralWidget(central);
    auto* root = new QVBoxLayout(central);
    root->setContentsMargins(4, 4, 4, 4);
    root->setSpacing(4);

    // ── top bar ─────────────────────────────────────────────
    auto* top = new QHBoxLayout;
    btnLoad_ = new QPushButton("  Load Image  ");
    btnLoad_->setFixedHeight(34);
    connect(btnLoad_, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    lblInfo_ = new QLabel("No image loaded — click \"Load Image\"");
    lblInfo_->setFont(QFont("Segoe UI", 10));
    top->addWidget(btnLoad_);
    top->addWidget(lblInfo_, 1);
    root->addLayout(top);

    // ── tabs ────────────────────────────────────────────────
    auto* tabs = new QTabWidget;
    tabs->addTab(buildNoiseFilterTab(), "Noise && Filter");
    tabs->addTab(buildEdgeTab(),        "Edge Detection");
    tabs->addTab(buildHistColorTab(),   "Histograms && Color");
    tabs->addTab(buildFFTTab(),         "Frequency Domain");
    tabs->addTab(buildHybridTab(),      "Hybrid Images");
    root->addWidget(tabs, 1);

    statusBar()->showMessage("Ready");
}

// ═══════════════════════════════════════════════════════════════════
//  Factory helpers
// ═══════════════════════════════════════════════════════════════════

ClickableImageLabel* MainWindow::makeImageLabel() {
    auto* lbl = new ClickableImageLabel;
    lbl->setAlignment(Qt::AlignCenter);
    lbl->setMinimumSize(60, 45);
    lbl->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    lbl->setStyleSheet("border:1px solid #888; background:#1a1a2e;");
    lbl->setScaledContents(false);

    // Double-click → open zoomable preview
    connect(lbl, &ClickableImageLabel::doubleClicked, this,
            [this, lbl]() { openPreview(lbl); });

    return lbl;
}

QPushButton* MainWindow::makeSaveButton(QLabel* label) {
    auto* btn = new QPushButton("Save");
    btn->setFixedHeight(22);
    btn->setStyleSheet("font-size:10px; padding:1px 8px;");
    btn->setVisible(false);  // hidden until an image is set
    connect(btn, &QPushButton::clicked, this,
            [this, label]() { saveImageForLabel(label); });
    labelSaveButtonMap_[label] = btn;
    return btn;
}

QLabel* MainWindow::makeMetricsLabel(const QString& prefix) {
    auto* lbl = new QLabel(prefix + "Metrics: —");
    lbl->setFont(QFont("Consolas", 9));
    lbl->setStyleSheet(
        "background:#1e1e2e; color:#cdd6f4; padding:4px; border-radius:3px;");
    return lbl;
}

void MainWindow::openPreview(QLabel* label) {
    if (!labelImageMap_.contains(label)) return;
    cv::Mat mat = labelImageMap_[label];
    if (mat.empty()) return;
    QImage qimg = matToQImage(mat);
    auto* dlg = new ZoomableImageDialog(qimg, mat, "Image Preview", this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->show();
}

void MainWindow::saveImageForLabel(QLabel* label) {
    if (!labelImageMap_.contains(label)) return;
    cv::Mat img = labelImageMap_[label];
    if (img.empty()) return;
    QString path = QFileDialog::getSaveFileName(
        this, "Save Image", "",
        "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All (*)");
    if (path.isEmpty()) return;
    cv::imwrite(path.toStdString(), img);
    statusBar()->showMessage("Saved: " + path);
}

// ═══════════════════════════════════════════════════════════════════
//  Build individual tabs
// ═══════════════════════════════════════════════════════════════════

QWidget* MainWindow::buildNoiseFilterTab() {
    auto* w = new QWidget;
    auto* lay = new QVBoxLayout(w);
    lay->setSpacing(4);

    // ── noise controls ──────────────────────────────────────
    auto* r1 = new QHBoxLayout;
    r1->addWidget(new QLabel("Noise:"));

    nfNoiseType_ = new QComboBox;
    nfNoiseType_->addItems({"gaussian", "uniform", "salt_pepper"});
    connect(nfNoiseType_, &QComboBox::currentTextChanged,
            this, &MainWindow::onNoiseTypeChanged);
    r1->addWidget(nfNoiseType_);

    r1->addWidget(new QLabel("Intensity:"));
    nfIntensity_ = new QDoubleSpinBox;
    nfIntensity_->setRange(0.0, 1.0);
    nfIntensity_->setSingleStep(0.05);
    nfIntensity_->setValue(0.3);
    r1->addWidget(nfIntensity_);

    nfLblMean_ = new QLabel("Mean:");
    r1->addWidget(nfLblMean_);
    nfMean_ = new QDoubleSpinBox;
    nfMean_->setRange(-255, 255);
    nfMean_->setValue(0);
    r1->addWidget(nfMean_);

    nfLblStd_ = new QLabel("Std:");
    r1->addWidget(nfLblStd_);
    nfStd_ = new QDoubleSpinBox;
    nfStd_->setRange(0.1, 255);
    nfStd_->setValue(100);
    r1->addWidget(nfStd_);

    nfLblRatio_ = new QLabel("S/P Ratio:");
    r1->addWidget(nfLblRatio_);
    nfSPRatio_ = new QDoubleSpinBox;
    nfSPRatio_->setRange(0.0, 1.0);
    nfSPRatio_->setSingleStep(0.1);
    nfSPRatio_->setValue(0.5);
    r1->addWidget(nfSPRatio_);

    auto* btnNoise = new QPushButton("Add Noise");
    connect(btnNoise, &QPushButton::clicked, this, &MainWindow::onApplyNoise);
    r1->addWidget(btnNoise);
    r1->addStretch();
    lay->addLayout(r1);

    onNoiseTypeChanged("gaussian");

    // ── filter controls ─────────────────────────────────────
    auto* r2 = new QHBoxLayout;
    r2->addWidget(new QLabel("Filter:"));
    nfFilterType_ = new QComboBox;
    nfFilterType_->addItems({"average", "gaussian", "median"});
    r2->addWidget(nfFilterType_);

    r2->addWidget(new QLabel("Kernel:"));
    nfKernel_ = new QSpinBox;
    nfKernel_->setRange(3, 15);
    nfKernel_->setSingleStep(2);
    nfKernel_->setValue(5);
    r2->addWidget(nfKernel_);

    auto* btnFilt = new QPushButton("Apply Filter");
    connect(btnFilt, &QPushButton::clicked, this, &MainWindow::onApplyFilter);
    r2->addWidget(btnFilt);
    r2->addStretch();
    lay->addLayout(r2);

    // ── image display area ──────────────────────────────────
    auto* imgRow = new QHBoxLayout;
    imgRow->setSpacing(4);
    for (int k = 0; k < 3; ++k) {
        auto* box = new QVBoxLayout;
        box->setSpacing(2);
        nfTitles_[k] = new QLabel;
        nfTitles_[k]->setAlignment(Qt::AlignCenter);
        nfTitles_[k]->setFont(QFont("Segoe UI", 8));
        nfImgs_[k] = makeImageLabel();
        box->addWidget(nfTitles_[k]);
        box->addWidget(nfImgs_[k], 1);
        box->addWidget(makeSaveButton(nfImgs_[k]));
        imgRow->addLayout(box, 1);
    }
    lay->addLayout(imgRow, 1);

    // ── metrics ─────────────────────────────────────────────
    nfNoiseMetrics_  = makeMetricsLabel("Noise — ");
    nfFilterMetrics_ = makeMetricsLabel("Filter — ");
    lay->addWidget(nfNoiseMetrics_);
    lay->addWidget(nfFilterMetrics_);

    return w;
}

QWidget* MainWindow::buildEdgeTab() {
    auto* w = new QWidget;
    auto* lay = new QVBoxLayout(w);
    lay->setSpacing(4);

    auto* ctrl = new QHBoxLayout;
    ctrl->addWidget(new QLabel("Method:"));
    edgeMethod_ = new QComboBox;
    edgeMethod_->addItems({"sobel", "roberts", "prewitt", "canny"});
    connect(edgeMethod_, &QComboBox::currentTextChanged,
            this, &MainWindow::onEdgeMethodChanged);
    ctrl->addWidget(edgeMethod_);

    lblCannyLo_ = new QLabel("Low:");
    ctrl->addWidget(lblCannyLo_);
    cannyLo_ = new QSpinBox;
    cannyLo_->setRange(1, 500);
    cannyLo_->setValue(50);
    ctrl->addWidget(cannyLo_);

    lblCannyHi_ = new QLabel("High:");
    ctrl->addWidget(lblCannyHi_);
    cannyHi_ = new QSpinBox;
    cannyHi_->setRange(1, 500);
    cannyHi_->setValue(150);
    ctrl->addWidget(cannyHi_);

    for (auto* wgt : {(QWidget*)lblCannyLo_, (QWidget*)cannyLo_,
                      (QWidget*)lblCannyHi_, (QWidget*)cannyHi_})
        wgt->setVisible(false);

    auto* btnEdge = new QPushButton("Detect Edges");
    connect(btnEdge, &QPushButton::clicked, this, &MainWindow::onApplyEdge);
    ctrl->addWidget(btnEdge);
    ctrl->addStretch();
    lay->addLayout(ctrl);

    // ── image display (up to 4) ─────────────────────────────
    auto* imgRow = new QHBoxLayout;
    imgRow->setSpacing(4);
    for (int k = 0; k < 4; ++k) {
        auto* box = new QVBoxLayout;
        box->setSpacing(2);
        edgeTitles_[k] = new QLabel;
        edgeTitles_[k]->setAlignment(Qt::AlignCenter);
        edgeTitles_[k]->setFont(QFont("Segoe UI", 8));
        edgeImgs_[k] = makeImageLabel();
        box->addWidget(edgeTitles_[k]);
        box->addWidget(edgeImgs_[k], 1);
        box->addWidget(makeSaveButton(edgeImgs_[k]));
        imgRow->addLayout(box, 1);
    }
    lay->addLayout(imgRow, 1);

    edgeMetrics_ = makeMetricsLabel();
    lay->addWidget(edgeMetrics_);
    return w;
}

QWidget* MainWindow::buildHistColorTab() {
    auto* w = new QWidget;
    auto* lay = new QVBoxLayout(w);
    lay->setSpacing(4);

    // ── buttons ─────────────────────────────────────────────
    auto* ctrl = new QHBoxLayout;

    auto* btnCh = new QPushButton("R / G / B  Channel Analysis");
    connect(btnCh, &QPushButton::clicked, this, &MainWindow::onShowChannels);
    ctrl->addWidget(btnCh);

    auto* btnEq = new QPushButton("Equalize");
    connect(btnEq, &QPushButton::clicked, this, &MainWindow::onShowEqualize);
    ctrl->addWidget(btnEq);

    auto* btnNorm = new QPushButton("Normalize");
    connect(btnNorm, &QPushButton::clicked, this, &MainWindow::onShowNormalize);
    ctrl->addWidget(btnNorm);

    auto* btnGray = new QPushButton("Grayscale");
    connect(btnGray, &QPushButton::clicked, this, &MainWindow::onShowGrayscale);
    ctrl->addWidget(btnGray);

    ctrl->addStretch();
    lay->addLayout(ctrl);

    // ── grid display area (3 rows × 5 cols) ─────────────────
    hcGrid_ = new QGridLayout;
    hcGrid_->setSpacing(3);
    for (int c = 0; c < HC_COLS; ++c)
        hcGrid_->setColumnStretch(c, 1);
    for (int r = 0; r < HC_ROWS; ++r) {
        hcGrid_->setRowStretch(r, 1);
        for (int c = 0; c < HC_COLS; ++c) {
            auto* box = new QVBoxLayout;
            box->setSpacing(1);
            hcTitles_[r][c] = new QLabel;
            hcTitles_[r][c]->setAlignment(Qt::AlignCenter);
            hcTitles_[r][c]->setFont(QFont("Segoe UI", 7));
            hcImgs_[r][c] = makeImageLabel();
            hcImgs_[r][c]->setVisible(false);
            hcTitles_[r][c]->setVisible(false);
            box->addWidget(hcTitles_[r][c]);
            box->addWidget(hcImgs_[r][c], 1);
            box->addWidget(makeSaveButton(hcImgs_[r][c]));
            hcGrid_->addLayout(box, r, c);
        }
    }
    lay->addLayout(hcGrid_, 1);

    return w;
}

QWidget* MainWindow::buildFFTTab() {
    auto* w = new QWidget;
    auto* lay = new QVBoxLayout(w);
    lay->setSpacing(4);

    auto* ctrl = new QHBoxLayout;
    ctrl->addWidget(new QLabel("Filter:"));
    fftType_ = new QComboBox;
    fftType_->addItems({"lowpass", "highpass"});
    ctrl->addWidget(fftType_);

    ctrl->addWidget(new QLabel("Cutoff:"));
    fftCutoff_ = new QSpinBox;
    fftCutoff_->setRange(1, 500);
    fftCutoff_->setValue(30);
    ctrl->addWidget(fftCutoff_);

    auto* btnFFT = new QPushButton("Apply FFT Filter");
    connect(btnFFT, &QPushButton::clicked, this, &MainWindow::onApplyFFT);
    ctrl->addWidget(btnFFT);
    ctrl->addStretch();
    lay->addLayout(ctrl);

    // ── display: original | spectrum | filtered ─────────────
    auto* imgRow = new QHBoxLayout;
    imgRow->setSpacing(4);
    for (int k = 0; k < 3; ++k) {
        auto* box = new QVBoxLayout;
        box->setSpacing(2);
        fftTitles_[k] = new QLabel;
        fftTitles_[k]->setAlignment(Qt::AlignCenter);
        fftTitles_[k]->setFont(QFont("Segoe UI", 8));
        fftImgs_[k] = makeImageLabel();
        box->addWidget(fftTitles_[k]);
        box->addWidget(fftImgs_[k], 1);
        box->addWidget(makeSaveButton(fftImgs_[k]));
        imgRow->addLayout(box, 1);
    }
    lay->addLayout(imgRow, 1);

    fftMetrics_ = makeMetricsLabel();
    lay->addWidget(fftMetrics_);
    return w;
}

QWidget* MainWindow::buildHybridTab() {
    auto* w = new QWidget;
    auto* lay = new QVBoxLayout(w);
    lay->setSpacing(4);

    // ── row 1: load Image B ─────────────────────────────────
    auto* r1 = new QHBoxLayout;
    btnLoadB_ = new QPushButton("Load Image B");
    connect(btnLoadB_, &QPushButton::clicked, this, &MainWindow::onLoadImageB);
    r1->addWidget(btnLoadB_);
    lblBInfo_ = new QLabel("No Image B loaded");
    r1->addWidget(lblBInfo_);
    r1->addStretch();
    lay->addLayout(r1);

    // ── row 2: hybrid controls ──────────────────────────────
    auto* r2 = new QHBoxLayout;
    r2->addWidget(new QLabel("Roles:"));
    hybridRoles_ = new QComboBox;
    hybridRoles_->addItems({
        "A = Low-pass,  B = High-pass",
        "A = High-pass,  B = Low-pass"
    });
    r2->addWidget(hybridRoles_);

    r2->addWidget(new QLabel("LP cutoff:"));
    hybridLP_ = new QSpinBox;
    hybridLP_->setRange(1, 300);
    hybridLP_->setValue(30);
    r2->addWidget(hybridLP_);

    r2->addWidget(new QLabel("HP cutoff:"));
    hybridHP_ = new QSpinBox;
    hybridHP_->setRange(1, 300);
    hybridHP_->setValue(30);
    r2->addWidget(hybridHP_);

    auto* btnHybrid = new QPushButton("Create Hybrid");
    connect(btnHybrid, &QPushButton::clicked, this, &MainWindow::onApplyHybrid);
    r2->addWidget(btnHybrid);
    r2->addStretch();
    lay->addLayout(r2);

    // ── display: A | B | hybrid ─────────────────────────────
    auto* imgRow = new QHBoxLayout;
    imgRow->setSpacing(4);
    for (int k = 0; k < 3; ++k) {
        auto* box = new QVBoxLayout;
        box->setSpacing(2);
        hybridTitles_[k] = new QLabel;
        hybridTitles_[k]->setAlignment(Qt::AlignCenter);
        hybridTitles_[k]->setFont(QFont("Segoe UI", 8));
        hybridImgs_[k] = makeImageLabel();
        box->addWidget(hybridTitles_[k]);
        box->addWidget(hybridImgs_[k], 1);
        box->addWidget(makeSaveButton(hybridImgs_[k]));
        imgRow->addLayout(box, 1);
    }
    lay->addLayout(imgRow, 1);

    hybridMetrics_ = makeMetricsLabel();
    lay->addWidget(hybridMetrics_);
    return w;
}

// ═══════════════════════════════════════════════════════════════════
//  Image Loading
// ═══════════════════════════════════════════════════════════════════

void MainWindow::onLoadImage() {
    QString path = QFileDialog::getOpenFileName(
        this, "Open Image", "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All (*)");
    if (path.isEmpty()) return;

    try {
        model_.load(path.toStdString());
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Error", e.what());
        return;
    }

    cachedNoisy_ = cv::Mat();
    cachedSpectrum_ = cv::Mat();

    cv::Mat img = model_.getOriginal();
    int h = img.rows, w = img.cols;
    lblInfo_->setText(QString("%1  —  %2×%3")
                          .arg(QFileInfo(path).fileName())
                          .arg(w).arg(h));
    statusBar()->showMessage("Loaded: " + path);
    showOriginalEverywhere();
}

cv::Mat MainWindow::requireImage() {
    if (!model_.isLoaded()) {
        QMessageBox::warning(this, "No Image", "Load an image first.");
        return {};
    }
    return model_.getOriginal();
}

void MainWindow::showOriginalEverywhere() {
    cv::Mat img = model_.getOriginal();

    // Tab 1
    showImageOnLabel(nfImgs_[0], img);
    nfTitles_[0]->setText("Original");
    for (int k = 1; k < 3; ++k) {
        showImageOnLabel(nfImgs_[k], cv::Mat());
        nfTitles_[k]->clear();
    }
    resetMetrics(nfNoiseMetrics_,  "Noise — ");
    resetMetrics(nfFilterMetrics_, "Filter — ");

    // Tab 2
    showImageOnLabel(edgeImgs_[0], img);
    edgeTitles_[0]->setText("Original");
    for (int k = 1; k < 4; ++k) {
        showImageOnLabel(edgeImgs_[k], cv::Mat());
        edgeTitles_[k]->clear();
        edgeImgs_[k]->setVisible(true);
    }
    resetMetrics(edgeMetrics_);

    // Tab 3 — hide everything
    for (int r = 0; r < HC_ROWS; ++r)
        for (int c = 0; c < HC_COLS; ++c) {
            showImageOnLabel(hcImgs_[r][c], cv::Mat());
            hcImgs_[r][c]->setVisible(false);
            hcTitles_[r][c]->clear();
            hcTitles_[r][c]->setVisible(false);
        }
    hcImgs_[0][0]->setVisible(true);
    hcTitles_[0][0]->setVisible(true);
    showImageOnLabel(hcImgs_[0][0], img);
    hcTitles_[0][0]->setText("Original");

    // Tab 4
    showImageOnLabel(fftImgs_[0], img);
    fftTitles_[0]->setText("Original");
    for (int k = 1; k < 3; ++k) {
        showImageOnLabel(fftImgs_[k], cv::Mat());
        fftTitles_[k]->clear();
    }
    resetMetrics(fftMetrics_);

    // Tab 5
    showImageOnLabel(hybridImgs_[0], img);
    hybridTitles_[0]->setText("Image A");
    for (int k = 1; k < 3; ++k) {
        showImageOnLabel(hybridImgs_[k], cv::Mat());
        hybridTitles_[k]->clear();
    }
    resetMetrics(hybridMetrics_);
}

// ═══════════════════════════════════════════════════════════════════
//  Display helpers
// ═══════════════════════════════════════════════════════════════════

void MainWindow::showImageOnLabel(QLabel* label, const cv::Mat& img) {
    if (img.empty()) {
        label->clear();
        // Clear original image on ClickableImageLabel
        if (auto* cil = qobject_cast<ClickableImageLabel*>(label))
            cil->clearOriginalImage();
        labelImageMap_.remove(label);
        if (labelSaveButtonMap_.contains(label))
            labelSaveButtonMap_[label]->setVisible(false);
        return;
    }

    // Store full-resolution image for saving / preview
    labelImageMap_[label] = img.clone();

    // Show the associated save button
    if (labelSaveButtonMap_.contains(label))
        labelSaveButtonMap_[label]->setVisible(true);

    QImage qimg = matToQImage(img);

    // If this is a ClickableImageLabel, use its resize-aware scaling
    if (auto* cil = qobject_cast<ClickableImageLabel*>(label)) {
        cil->setOriginalImage(qimg);
    } else {
        // Fallback for plain QLabel
        QSize sz = label->size();
        if (sz.width() < 20 || sz.height() < 20)
            sz = QSize(300, 200);
        QPixmap pix = QPixmap::fromImage(qimg)
                          .scaled(sz, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        label->setPixmap(pix);
    }
}

void MainWindow::setMetricsText(QLabel* label, const cv::Mat& orig,
                                const cv::Mat& proc, const QString& prefix) {
    cv::Mat o = orig, p = proc;
    if (o.channels() == 3 && p.channels() == 1)
        o = ColorProcessor::toGrayscale(o);
    else if (o.channels() == 1 && p.channels() == 3)
        p = ColorProcessor::toGrayscale(p);

    auto m = metrics_.computeAll(o, p);
    label->setText(
        QString("%1MSE: %2  |  PSNR: %3 dB  |  SNR: %4 dB")
            .arg(prefix)
            .arg(m["MSE"],  0, 'f', 2)
            .arg(m["PSNR"], 0, 'f', 2)
            .arg(m["SNR"],  0, 'f', 2));
}

void MainWindow::resetMetrics(QLabel* label, const QString& prefix) {
    label->setText(prefix + "Metrics: —");
}

// ═══════════════════════════════════════════════════════════════════
//  cv::Mat → QImage conversion
// ═══════════════════════════════════════════════════════════════════

QImage MainWindow::matToQImage(const cv::Mat& mat) {
    if (mat.empty()) return {};
    if (mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows,
                      static_cast<int>(rgb.step),
                      QImage::Format_RGB888).copy();
    }
    if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows,
                      static_cast<int>(mat.step),
                      QImage::Format_Grayscale8).copy();
    }
    return {};
}

// ═══════════════════════════════════════════════════════════════════
//  Histogram / CDF rendering utilities
// ═══════════════════════════════════════════════════════════════════

cv::Mat MainWindow::renderHistogramImage(const std::vector<int>& counts,
                                          const cv::Scalar& color,
                                          int h, int w) {
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    int maxVal = *std::max_element(counts.begin(), counts.end());
    if (maxVal == 0) return canvas;

    double binW = static_cast<double>(w) / 256.0;
    for (int i = 0; i < 256; ++i) {
        int barH = cvRound(static_cast<double>(counts[i]) / maxVal * (h - 10));
        cv::Point p1(cvRound(i * binW), h - 1);
        cv::Point p2(cvRound((i + 1) * binW) - 1, h - 1 - barH);
        cv::rectangle(canvas, p1, p2, color, cv::FILLED);
    }
    return canvas;
}

cv::Mat MainWindow::renderCDFImage(const std::vector<double>& cdf,
                                    const cv::Scalar& color,
                                    int h, int w) {
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    double binW = static_cast<double>(w) / 256.0;
    int thickness = std::max(2, h / 100);

    for (int i = 1; i < 256; ++i) {
        cv::Point p1(cvRound((i - 1) * binW),
                     h - 1 - cvRound(cdf[i - 1] * (h - 10)));
        cv::Point p2(cvRound(i * binW),
                     h - 1 - cvRound(cdf[i] * (h - 10)));
        cv::line(canvas, p1, p2, color, thickness);
    }
    return canvas;
}

cv::Mat MainWindow::renderCDFOverlay(
        const std::vector<std::pair<std::vector<double>, cv::Scalar>>& ds,
        int h, int w) {
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    double binW = static_cast<double>(w) / 256.0;
    int thickness = std::max(2, h / 100);

    for (const auto& [cdf, color] : ds) {
        for (int i = 1; i < 256; ++i) {
            cv::Point p1(cvRound((i - 1) * binW),
                         h - 1 - cvRound(cdf[i - 1] * (h - 10)));
            cv::Point p2(cvRound(i * binW),
                         h - 1 - cvRound(cdf[i] * (h - 10)));
            cv::line(canvas, p1, p2, color, thickness);
        }
    }
    return canvas;
}
