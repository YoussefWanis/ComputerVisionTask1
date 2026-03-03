/**
 * @file MainWindow.h
 * @brief Declaration of the MainWindow class and its helper
 *        ClickableImageLabel widget.
 *
 * MainWindow is the central GUI class for the CV Task 1 application.
 * It contains five processing tabs:
 *   1. Noise & Filter       — add noise and apply spatial filters.
 *   2. Edge Detection        — Sobel, Roberts, Prewitt, Canny.
 *   3. Histograms & Colour   — channel analysis, equalisation, normalisation.
 *   4. Frequency Domain      — FFT-based low/high-pass filtering.
 *   5. Hybrid Images         — combine two images in frequency domain.
 *
 * Each tab has its own buildXxxTab() factory method and one or more
 * slot methods that perform the processing and update the display.
 * Slot implementations are split across multiple .cpp files
 * (MainWindow.cpp, MainWindow_NoiseFilter.cpp, etc.) for readability.
 */

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

/**
 * @class ClickableImageLabel
 * @brief A QLabel subclass that emits a doubleClicked() signal and
 *        supports automatic resize-aware pixmap scaling.
 *
 * Used as the image display widget in every tab.  Stores the original
 * full-resolution QImage and re-scales it whenever the label is
 * resized, preserving the aspect ratio with smooth interpolation.
 *
 * Double-clicking the label opens a ZoomableImageDialog for
 * detailed inspection.
 */
class ClickableImageLabel : public QLabel {
    Q_OBJECT
public:
    /**
     * @brief Construct a ClickableImageLabel.
     * @param parent  Parent widget (typically a tab layout).
     */
    explicit ClickableImageLabel(QWidget* parent = nullptr)
        : QLabel(parent) {}

    /**
     * @brief Store the original full-resolution image and rescale.
     *
     * The image is kept internally so that every subsequent resize
     * event can re-generate a properly scaled pixmap without quality
     * loss from repeated re-sampling.
     *
     * @param img  Full-resolution QImage to display.
     */
    void setOriginalImage(const QImage& img) {
        originalImage_ = img;
        rescalePixmap();
    }

    /** @brief Clear the stored image and pixmap. */
    void clearOriginalImage() {
        originalImage_ = QImage();
    }

signals:
    /** @brief Emitted when the user double-clicks this label. */
    void doubleClicked();

protected:
    /** @brief Forward double-click events as a signal. */
    void mouseDoubleClickEvent(QMouseEvent* event) override {
        emit doubleClicked();
        QLabel::mouseDoubleClickEvent(event);
    }

    /** @brief Re-scale the stored image to fit the new label size. */
    void resizeEvent(QResizeEvent* event) override {
        QLabel::resizeEvent(event);
        rescalePixmap();
    }

private:
    /**
     * @brief Scale the stored QImage to the label's current size.
     *
     * Uses Qt::KeepAspectRatio and Qt::SmoothTransformation for
     * high-quality downscaling.  No-op if no image is stored or
     * the label is too small.
     */
    void rescalePixmap() {
        if (originalImage_.isNull()) return;
        QSize sz = size();
        if (sz.width() < 10 || sz.height() < 10) return;
        QPixmap pix = QPixmap::fromImage(originalImage_)
                          .scaled(sz, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        setPixmap(pix);
    }

    QImage originalImage_;   ///< Full-resolution source image for rescaling.
};

/**
 * @class MainWindow
 * @brief Main application window with five image-processing tabs.
 *
 * Manages:
 *   - Image loading via QFileDialog.
 *   - Processing through the various Processor classes.
 *   - Display of original, intermediate, and processed images.
 *   - Quality metrics (MSE, PSNR, SNR) shown below each tab.
 *   - Image saving and zoomable preview dialogs.
 *
 * Slot implementations are split across several .cpp files:
 *   - MainWindow.cpp            — constructor, UI build, display helpers.
 *   - MainWindow_NoiseFilter.cpp — Tab 1 slots.
 *   - MainWindow_Edge.cpp       — Tab 2 slots.
 *   - MainWindow_Histogram.cpp  — Tab 3 slots.
 *   - MainWindow_FFT.cpp        — Tab 4 slots.
 *   - MainWindow_Hybrid.cpp     — Tab 5 slots.
 */
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    /**
     * @brief Construct the main window and build its entire UI.
     * @param parent  Parent widget (nullptr for a top-level window).
     */
    explicit MainWindow(QWidget* parent = nullptr);

    /** @brief Default destructor. */
    ~MainWindow() override = default;

    // ── Display helpers (public for helper free-functions) ───

    /**
     * @brief Display an OpenCV image on a QLabel.
     *
     * Stores a full-resolution clone in labelImageMap_ and sets the
     * pixmap on the label (with automatic scaling for ClickableImageLabel).
     * Passing an empty Mat clears the label.
     *
     * @param label  Target QLabel (typically a ClickableImageLabel).
     * @param img    Image to display (CV_8UC1 or CV_8UC3), or empty to clear.
     */
    void showImageOnLabel(QLabel* label, const cv::Mat& img);

    /**
     * @brief Compute and display quality metrics on a label.
     *
     * Calculates MSE, PSNR, and SNR between the original and processed
     * images, handling channel-count mismatches by converting to grayscale.
     *
     * @param label   QLabel where the metrics text is shown.
     * @param orig    Original (reference) image.
     * @param proc    Processed (potentially degraded) image.
     * @param prefix  Optional text prepended to the metrics line.
     */
    void setMetricsText(QLabel* label, const cv::Mat& orig,
                        const cv::Mat& proc, const QString& prefix = "");

    /**
     * @brief Reset a metrics label to its default "Metrics: —" text.
     * @param label   QLabel to reset.
     * @param prefix  Optional prefix (e.g. "Noise — ").
     */
    void resetMetrics(QLabel* label, const QString& prefix = "");

    /**
     * @brief Show the original loaded image on every tab's first panel.
     *
     * Called after a new image is loaded to initialise all display slots.
     */
    void showOriginalEverywhere();

    /**
     * @brief Return the original image, or show a warning and return empty.
     * @return  Deep copy of the loaded image, or an empty Mat.
     */
    cv::Mat requireImage();

    // ── Image conversion / rendering ────────────────────────

    /**
     * @brief Convert an OpenCV Mat (BGR or grayscale) to a QImage.
     * @param mat  Input Mat (CV_8UC3 or CV_8UC1).
     * @return     QImage copy, or empty QImage if conversion fails.
     */
    static QImage matToQImage(const cv::Mat& mat);

    /**
     * @brief Render a 256-bin histogram as a bar-chart image.
     * @param counts  256-element vector of bin counts.
     * @param color   Bar colour in BGR.
     * @param h       Canvas height in pixels.
     * @param w       Canvas width in pixels.
     * @return        CV_8UC3 histogram image.
     */
    static cv::Mat renderHistogramImage(const std::vector<int>& counts,
                                        const cv::Scalar& color,
                                        int h = 512, int w = 768);

    /**
     * @brief Render a single-channel CDF as a line-chart image.
     * @param cdf    256-element normalised CDF [0, 1].
     * @param color  Line colour in BGR.
     * @param h      Canvas height in pixels.
     * @param w      Canvas width in pixels.
     * @return       CV_8UC3 CDF image.
     */
    static cv::Mat renderCDFImage(const std::vector<double>& cdf,
                                   const cv::Scalar& color,
                                   int h = 512, int w = 768);

    /**
     * @brief Render multiple CDF curves overlaid on a single canvas.
     * @param ds  Vector of (CDF, colour) pairs.
     * @param h   Canvas height in pixels.
     * @param w   Canvas width in pixels.
     * @return    CV_8UC3 overlay image.
     */
    static cv::Mat renderCDFOverlay(
            const std::vector<std::pair<std::vector<double>, cv::Scalar>>& ds,
            int h = 512, int w = 768);

    // ── Map of labels to their full-resolution cv::Mat ──────
    /** Label → full-res image map (used for saving / preview). */
    QHash<QLabel*, cv::Mat> labelImageMap_;

    // ── Map of labels to their associated Save buttons ──────
    /** Label → Save button map (visibility toggled with image). */
    QHash<QLabel*, QPushButton*> labelSaveButtonMap_;

private slots:
    // ── General ─────────────────────────────────────────────
    /** @brief Slot: open a file dialog and load an image. */
    void onLoadImage();

    // ── Tab 1: Noise & Filter ───────────────────────────────
    /** @brief Slot: show/hide noise-type-specific controls. */
    void onNoiseTypeChanged(const QString& type);
    /** @brief Slot: generate a noisy version of the loaded image. */
    void onApplyNoise();
    /** @brief Slot: apply a spatial filter to the noisy image. */
    void onApplyFilter();

    // ── Tab 2: Edge Detection ───────────────────────────────
    /** @brief Slot: show/hide Canny threshold controls. */
    void onEdgeMethodChanged(const QString& method);
    /** @brief Slot: run edge detection on the loaded image. */
    void onApplyEdge();

    // ── Tab 3: Histograms & Colour ──────────────────────────
    /** @brief Slot: equalise the histogram and show before/after. */
    void onShowEqualize();
    /** @brief Slot: normalise the image and show before/after. */
    void onShowNormalize();

    // ── Tab 4: Frequency Domain ─────────────────────────────
    /** @brief Slot: apply FFT low/high-pass filter and display results. */
    void onApplyFFT();

    // ── Tab 5: Hybrid Images ────────────────────────────────
    /** @brief Slot: load a second image for hybrid creation. */
    void onLoadImageB();
    /** @brief Slot: create and display the hybrid image. */
    void onApplyHybrid();

private:
    // ── UI construction ─────────────────────────────────────

    /** @brief Build the entire main-window UI (top bar + tabs). */
    void buildUI();

    /** @brief Build and return Tab 1: Noise & Filter. */
    QWidget* buildNoiseFilterTab();
    /** @brief Build and return Tab 2: Edge Detection. */
    QWidget* buildEdgeTab();
    /** @brief Build and return Tab 3: Histograms & Colour. */
    QWidget* buildHistColorTab();
    /** @brief Build and return Tab 4: Frequency Domain. */
    QWidget* buildFFTTab();
    /** @brief Build and return Tab 5: Hybrid Images. */
    QWidget* buildHybridTab();

    /**
     * @brief Create a ClickableImageLabel ready for image display.
     *
     * The label expands to fill available space, has a dark background
     * with a thin border, and double-clicking opens a zoomable preview.
     *
     * @return  Newly created ClickableImageLabel.
     */
    ClickableImageLabel* makeImageLabel();

    /**
     * @brief Create a "Save" button wired to save the image on the
     *        given label to disk.
     *
     * @param label  The image label whose content will be saved.
     * @return       Newly created QPushButton (initially hidden).
     */
    QPushButton* makeSaveButton(QLabel* label);

    /**
     * @brief Create a styled QLabel for displaying quality metrics.
     * @param prefix  Optional prefix text (e.g. "Noise — ").
     * @return        Newly created metrics QLabel.
     */
    QLabel* makeMetricsLabel(const QString& prefix = "");

    /**
     * @brief Open a ZoomableImageDialog to preview a label's image.
     * @param label  The label whose stored image should be previewed.
     */
    void openPreview(QLabel* label);

    /**
     * @brief Save the full-resolution image stored for a label to disk.
     * @param label  The label whose stored image should be saved.
     */
    void saveImageForLabel(QLabel* label);

    // ── Processing state ────────────────────────────────────
    ImageModel             model_;       ///< Loaded image + results cache.
    NoiseProcessor         noiseProc_;   ///< Noise generation engine.
    FilterProcessor        filterProc_;  ///< Spatial filter engine.
    EdgeDetectorProcessor  edgeProc_;    ///< Edge detection engine.
    HistogramProcessor     histProc_;    ///< Histogram processing engine.
    FFTProcessor           fftProc_;     ///< Frequency-domain filter engine.
    MetricsCalculator      metrics_;     ///< Image quality metrics calculator.

    cv::Mat cachedNoisy_;      ///< Last generated noisy image (for filtering).
    cv::Mat cachedSpectrum_;   ///< Cached FFT magnitude spectrum for display.
    cv::Mat imageBFull_;       ///< Full-resolution Image B (for hybrid tab).
    QString imageBPath_;       ///< File path of the loaded Image B.

    // ── Top bar widgets ─────────────────────────────────────
    QPushButton* btnLoad_ = nullptr;   ///< "Load Image" button.
    QLabel*      lblInfo_ = nullptr;   ///< Filename / resolution info label.

    // ── Tab 1: Noise & Filter widgets ───────────────────────
    QComboBox*      nfNoiseType_  = nullptr;   ///< Noise type combo box.
    QDoubleSpinBox* nfIntensity_  = nullptr;   ///< Noise intensity spin box.
    QLabel*         nfLblMean_    = nullptr;   ///< "Mean:" label (Gaussian).
    QDoubleSpinBox* nfMean_       = nullptr;   ///< Gaussian mean spin box.
    QLabel*         nfLblStd_     = nullptr;   ///< "Std:" label (Gaussian).
    QDoubleSpinBox* nfStd_        = nullptr;   ///< Gaussian std-dev spin box.
    QLabel*         nfLblRatio_   = nullptr;   ///< "S/P Ratio:" label.
    QDoubleSpinBox* nfSPRatio_    = nullptr;   ///< Salt/pepper ratio spin box.
    QComboBox*      nfFilterType_ = nullptr;   ///< Filter type combo box.
    QSpinBox*       nfKernel_     = nullptr;   ///< Filter kernel size spin box.
    QLabel*         nfImgs_[3]    = {};        ///< Image panels: Original, Noisy, Filtered.
    QLabel*         nfTitles_[3]  = {};        ///< Titles for the three panels.
    QLabel*         nfNoiseMetrics_  = nullptr; ///< Noise quality metrics label.
    QLabel*         nfFilterMetrics_ = nullptr; ///< Filter quality metrics label.

    // ── Tab 2: Edge Detection widgets ───────────────────────
    QComboBox* edgeMethod_  = nullptr;   ///< Edge method combo box.
    QLabel*    lblCannyLo_  = nullptr;   ///< "Low:" label for Canny.
    QSpinBox*  cannyLo_     = nullptr;   ///< Canny low-threshold spin box.
    QLabel*    lblCannyHi_  = nullptr;   ///< "High:" label for Canny.
    QSpinBox*  cannyHi_     = nullptr;   ///< Canny high-threshold spin box.
    QLabel*    edgeImgs_[4]   = {};      ///< Image panels: Original, X, Y, Combined.
    QLabel*    edgeTitles_[4] = {};      ///< Titles for the four panels.
    QLabel*    edgeMetrics_   = nullptr; ///< Edge quality metrics label.

    // ── Tab 3: Histograms & Colour widgets ──────────────────
    static constexpr int HC_ROWS = 3;    ///< Grid rows in the histogram tab.
    static constexpr int HC_COLS = 5;    ///< Grid columns in the histogram tab.
    QLabel*      hcImgs_[HC_ROWS][HC_COLS]   = {};   ///< Image grid cells.
    QLabel*      hcTitles_[HC_ROWS][HC_COLS]  = {};  ///< Title labels for grid cells.
    QGridLayout* hcGrid_ = nullptr;      ///< Grid layout managing the cells.

    // ── Tab 4: Frequency Domain widgets ─────────────────────
    QComboBox* fftType_   = nullptr;     ///< Filter type combo (lowpass/highpass).
    QSpinBox*  fftCutoff_ = nullptr;     ///< Cutoff radius spin box.
    QLabel*    fftImgs_[3]   = {};       ///< Panels: Original, Spectrum, Filtered.
    QLabel*    fftTitles_[3] = {};       ///< Titles for the three panels.
    QLabel*    fftMetrics_   = nullptr;  ///< FFT quality metrics label.

    // ── Tab 5: Hybrid Images widgets ────────────────────────
    QPushButton* btnLoadB_      = nullptr;   ///< "Load Image B" button.
    QLabel*      lblBInfo_      = nullptr;   ///< Image B info label.
    QComboBox*   hybridRoles_   = nullptr;   ///< LP/HP role assignment combo.
    QSpinBox*    hybridLP_      = nullptr;   ///< LP cutoff radius spin box.
    QSpinBox*    hybridHP_      = nullptr;   ///< HP cutoff radius spin box.
    QLabel*      hybridImgs_[3]   = {};      ///< Panels: Image A, Image B, Hybrid.
    QLabel*      hybridTitles_[3] = {};      ///< Titles for the three panels.
    QLabel*      hybridMetrics_   = nullptr; ///< Hybrid quality metrics label.
};

#endif // MAINWINDOW_H
