#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QTabWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QStatusBar>
#include <QFont>
#include <QMenu>
#include <QMouseEvent>
#include <QResizeEvent>
#include <opencv2/opencv.hpp>

#include "models/ImageModel.h"
#include "processors/ColorProcessor.h"
#include "processors/NoiseProcessor.h"
#include "processors/FilterProcessor.h"
#include "processors/EdgeDetectorProcessor.h"
#include "processors/HistogramProcessor.h"
#include "processors/FFTProcessor.h"
#include "processors/HybridProcessor.h"
#include "metrics/MetricsCalculator.h"

class MainWindow;

// ── Clickable QLabel that emits doubleClicked() ─────────────
class ClickableImageLabel : public QLabel {
    Q_OBJECT
public:
    explicit ClickableImageLabel(QWidget* parent = nullptr)
        : QLabel(parent) {}

    /** Store the original full-size QImage for re-scaling on resize. */
    void setOriginalImage(const QImage& img) {
        originalImage_ = img;
        rescalePixmap();
    }

    void clearOriginalImage() {
        originalImage_ = QImage();
    }

signals:
    void doubleClicked();

protected:
    void mouseDoubleClickEvent(QMouseEvent* event) override {
        emit doubleClicked();
        QLabel::mouseDoubleClickEvent(event);
    }

    void resizeEvent(QResizeEvent* event) override {
        QLabel::resizeEvent(event);
        rescalePixmap();
    }

private:
    void rescalePixmap() {
        if (originalImage_.isNull()) return;
        QSize sz = size();
        if (sz.width() < 10 || sz.height() < 10) return;
        QPixmap pix = QPixmap::fromImage(originalImage_)
                          .scaled(sz, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        setPixmap(pix);
    }

    QImage originalImage_;
};

/**
 * MainWindow — main application window with 5 tabs matching the Python GUI.
 *
 *  1. Noise & Filter
 *  2. Edge Detection
 *  3. Histograms & Color
 *  4. Frequency Domain
 *  5. Hybrid Images
 */
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override = default;

    // ── Display helpers (public for helper free-functions) ───
    void showImageOnLabel(QLabel* label, const cv::Mat& img);
    void setMetricsText(QLabel* label, const cv::Mat& orig,
                        const cv::Mat& proc, const QString& prefix = "");
    void resetMetrics(QLabel* label, const QString& prefix = "");
    void showOriginalEverywhere();
    cv::Mat requireImage();

    // ── Image conversion / rendering ────────────────────────
    static QImage matToQImage(const cv::Mat& mat);
    static cv::Mat renderHistogramImage(const std::vector<int>& counts,
                                        const cv::Scalar& color,
                                        int h = 512, int w = 768);
    static cv::Mat renderCDFImage(const std::vector<double>& cdf,
                                   const cv::Scalar& color,
                                   int h = 512, int w = 768);
    static cv::Mat renderCDFOverlay(
            const std::vector<std::pair<std::vector<double>, cv::Scalar>>& ds,
            int h = 512, int w = 768);

    // ── Map of labels to their full-resolution cv::Mat ──────
    QHash<QLabel*, cv::Mat> labelImageMap_;

    // ── Map of labels to their associated Save buttons ──────
    QHash<QLabel*, QPushButton*> labelSaveButtonMap_;

private slots:
    // General
    void onLoadImage();

    // Tab 1
    void onNoiseTypeChanged(const QString& type);
    void onApplyNoise();
    void onApplyFilter();

    // Tab 2
    void onEdgeMethodChanged(const QString& method);
    void onApplyEdge();

    // Tab 3
    void onShowChannels();
    void onShowEqualize();
    void onShowNormalize();
    void onShowGrayscale();

    // Tab 4
    void onApplyFFT();

    // Tab 5
    void onLoadImageB();
    void onApplyHybrid();

private:
    // ── UI construction ─────────────────────────────────────
    void buildUI();
    QWidget* buildNoiseFilterTab();
    QWidget* buildEdgeTab();
    QWidget* buildHistColorTab();
    QWidget* buildFFTTab();
    QWidget* buildHybridTab();

    /** Create a ClickableImageLabel ready for image display.
     *  Expands to fill available space; double-click opens preview. */
    ClickableImageLabel* makeImageLabel();

    /** Create a "Save" button wired to save the image on the given label. */
    QPushButton* makeSaveButton(QLabel* label);

    /** Create a styled metrics QLabel. */
    QLabel* makeMetricsLabel(const QString& prefix = "");

    /** Open the zoomable preview dialog for a label. */
    void openPreview(QLabel* label);

    /** Save the full-res image stored for a label. */
    void saveImageForLabel(QLabel* label);

    // ── Processing state ────────────────────────────────────
    ImageModel             model_;
    NoiseProcessor         noiseProc_;
    FilterProcessor        filterProc_;
    EdgeDetectorProcessor  edgeProc_;
    HistogramProcessor     histProc_;
    FFTProcessor           fftProc_;
    MetricsCalculator      metrics_;

    cv::Mat cachedNoisy_;
    cv::Mat cachedSpectrum_;
    cv::Mat imageBFull_;
    QString imageBPath_;

    // ── Top bar ─────────────────────────────────────────────
    QPushButton* btnLoad_ = nullptr;
    QLabel*      lblInfo_ = nullptr;

    // ── Tab 1: Noise & Filter ───────────────────────────────
    QComboBox*      nfNoiseType_  = nullptr;
    QDoubleSpinBox* nfIntensity_  = nullptr;
    QLabel*         nfLblMean_    = nullptr;
    QDoubleSpinBox* nfMean_       = nullptr;
    QLabel*         nfLblStd_     = nullptr;
    QDoubleSpinBox* nfStd_        = nullptr;
    QLabel*         nfLblRatio_   = nullptr;
    QDoubleSpinBox* nfSPRatio_    = nullptr;
    QComboBox*      nfFilterType_ = nullptr;
    QSpinBox*       nfKernel_     = nullptr;
    QLabel*         nfImgs_[3]    = {};
    QLabel*         nfTitles_[3]  = {};
    QLabel*         nfNoiseMetrics_  = nullptr;
    QLabel*         nfFilterMetrics_ = nullptr;

    // ── Tab 2: Edge Detection ───────────────────────────────
    QComboBox* edgeMethod_  = nullptr;
    QLabel*    lblCannyLo_  = nullptr;
    QSpinBox*  cannyLo_     = nullptr;
    QLabel*    lblCannyHi_  = nullptr;
    QSpinBox*  cannyHi_     = nullptr;
    QLabel*    edgeImgs_[4]   = {};
    QLabel*    edgeTitles_[4] = {};
    QLabel*    edgeMetrics_   = nullptr;

    // ── Tab 3: Histograms & Color ───────────────────────────
    static constexpr int HC_ROWS = 3;
    static constexpr int HC_COLS = 5;
    QLabel*      hcImgs_[HC_ROWS][HC_COLS]   = {};
    QLabel*      hcTitles_[HC_ROWS][HC_COLS]  = {};
    QGridLayout* hcGrid_ = nullptr;

    // ── Tab 4: Frequency Domain ─────────────────────────────
    QComboBox* fftType_   = nullptr;
    QSpinBox*  fftCutoff_ = nullptr;
    QLabel*    fftImgs_[3]   = {};
    QLabel*    fftTitles_[3] = {};
    QLabel*    fftMetrics_   = nullptr;

    // ── Tab 5: Hybrid Images ────────────────────────────────
    QPushButton* btnLoadB_      = nullptr;
    QLabel*      lblBInfo_      = nullptr;
    QComboBox*   hybridRoles_   = nullptr;
    QSpinBox*    hybridLP_      = nullptr;
    QSpinBox*    hybridHP_      = nullptr;
    QLabel*      hybridImgs_[3]   = {};
    QLabel*      hybridTitles_[3] = {};
    QLabel*      hybridMetrics_   = nullptr;
};

#endif // MAINWINDOW_H
