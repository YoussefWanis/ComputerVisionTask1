#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QTabWidget>
#include <QHBoxLayout>
#include <QButtonGroup>
#include <QRadioButton>
#include <QWidget>
#include <opencv2/opencv.hpp>
#include <vector>
#include "FrequencyDomain.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private slots:
    // File
    void handleLoad();
    void handleSave();
    void handleReset();

    // Noise
    void handleUniformNoise();
    void handleGaussianNoise();
    void handleSaltPepperNoise();
    void handleAllNoise();

    // Filters
    void handleAverageFilter();
    void handleGaussianFilter();
    void handleMedianFilter();
    void handleAllFilters();

    // Edge
    void handleSobel();
    void handleRoberts();
    void handlePrewitt();
    void handleCanny();
    void handleAllEdges();

    // Histogram
    void handleShowHistogramGray();
    void handleShowHistogramRGB();
    void handleEqualize();
    void handleNormalize();

    // Threshold
    void handleGlobalThreshold();
    void handleLocalThreshold();

    // Frequency (cached FFT)
    void handleLowPass();
    void handleHighPass();

    // Hybrid
    void loadSecondImage();
    void handleHybrid();

private:
    void setupUI();
    QWidget* createTab(const QString& title, const QStringList& actions);

    // Display helpers
    void updateDisplay(const cv::Mat& mat);
    void updateDisplay(const std::vector<cv::Mat>& images, const QStringList& titles = QStringList());
    QPixmap scaleAndConvert(const cv::Mat& mat, int maxWidth, int maxHeight);

    // FFT cache for main image
    void ensureFFTCached();
    FFTData cachedFFT;
    bool fftValid;

    // FFT cache for second image
    void ensureSecondFFTCached();
    FFTData cachedFFTSecond;
    bool secondFFTValid;

    // Hybrid mode selection (true = Image1 low, Image2 high; false = Image1 high, Image2 low)
    bool hybridMode;  // true = first low, second high (default)

    // UI elements for hybrid tab (to access radio buttons)
    QRadioButton *radioFirstLow;
    QRadioButton *radioFirstHigh;

    // Display area
    QWidget *displayContainer;
    QHBoxLayout *displayLayout;
    QTabWidget *tabs;

    // Image data
    cv::Mat originalFull;      // original loaded image (full size)
    cv::Mat originalResized;   // resized version for processing
    cv::Mat processedMat;      // current result to display
    cv::Mat secondFull;        // second image (full size)
    cv::Mat secondResized;     // resized second image
};

#endif // MAINWINDOW_H