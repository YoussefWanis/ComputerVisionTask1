#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QSlider>
#include <QRadioButton>
#include <QButtonGroup>
#include <QComboBox>
#include <QGroupBox>
#include <QGridLayout>
#include <QMap>
#include <opencv2/opencv.hpp>
#include "FrequencyDomain.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

private slots:
    // File actions
    void handleLoad();
    void handleSave();

    // Noise tab – processing (called on slider release)
    void updateUniformNoise();
    void updateGaussianNoise();
    void updateSaltPepperNoise();
    // Noise tab – label updates (called on value change)
    void updateUniformLabels();
    void updateGaussianLabels();
    void updateSPLabels();

    // Filters tab
    void updateAverageFilter();
    void updateGaussianFilter();
    void updateMedianFilter();
    void updateAvgLabels();
    void updateGaussFilterLabels();
    void updateMedianLabels();

    // Edge tab
    void updateSobel();
    void updateSobelX();
    void updateSobelY();
    void updateRoberts();
    void updatePrewitt();
    void updateCanny();
    void updateCannyLabels();

    // Histogram tab – RGB channels
    void updateRGBChannels();

    // Histogram tab – Equalize / Normalize
    void updateEqualize();
    void updateNormalize();

    // Threshold tab
    void updateGlobalThreshold();
    void updateLocalThreshold();
    void updateGlobalLabels();
    void updateLocalLabels();

    // Frequency tab
    void updateLowPass();
    void updateHighPass();
    void updateLowPassLabels();
    void updateHighPassLabels();

    // Hybrid tab
    void loadSecondImage();
    void updateHybrid();
    void updateHybridLabels();

    // Noise+Filter tab
    void updateNoiseFilter();
    void onNoiseTypeChanged(int index);
    void onFilterTypeChanged(int index);
    void updateNoiseFilterLabels();

    // Dark mode toggle
    void toggleDarkMode(bool checked);

private:
    void setupUI();
    void createToolBar();
    void applyTheme(bool dark);
    void setupImageLabel(QLabel* label);
    void showImage(QLabel* label, const cv::Mat& mat);
    void computeMetrics(const cv::Mat& original, const cv::Mat& processed,
                        double& mse, double& psnr, double& snr);
    void showEnlargedImage(const cv::Mat& img);

    // Original image
    cv::Mat originalFull;
    cv::Mat originalResized;

    // Second image for hybrid
    cv::Mat secondFull;
    cv::Mat secondResized;
    bool secondValid;

    // FFT cache
    FFTData cachedFFT;
    bool fftValid;
    FFTData cachedFFTSecond;
    bool secondFFTValid;

    // Map for double-click
    QMap<QLabel*, cv::Mat*> imageMap;

    // --- Noise tab ---
    QLabel *noiseOriginalLabel;
    QLabel *uniformLabel, *gaussianLabel, *spLabel;
    QSlider *uniformLowSlider, *uniformHighSlider;
    QSlider *gaussianMeanSlider, *gaussianStdSlider;
    QSlider *spProbSlider;
    QLabel *uniformLowVal, *uniformHighVal;
    QLabel *gaussianMeanVal, *gaussianStdVal;
    QLabel *spProbVal;
    cv::Mat uniformResult, gaussianResult, spResult;

    // --- Filters tab ---
    QLabel *filterOriginalLabel;
    QLabel *averageLabel, *gaussFilterLabel, *medianLabel;
    QSlider *avgKernelSlider, *gaussKernelSlider, *gaussSigmaSlider, *medianKernelSlider;
    QLabel *avgKernelVal, *gaussKernelVal, *gaussSigmaVal, *medianKernelVal;
    cv::Mat averageResult, gaussFilterResult, medianResult;

    // --- Edge tab ---
    QLabel *edgeOriginalLabel;
    QLabel *sobelLabel, *sobelXLabel, *sobelYLabel, *robertsLabel, *prewittLabel, *cannyLabel;
    QSlider *cannyLowSlider, *cannyHighSlider;
    QLabel *cannyLowVal, *cannyHighVal;
    cv::Mat sobelResult, sobelXResult, sobelYResult, robertsResult, prewittResult, cannyResult;

    // --- Histogram tab – RGB channels ---
    QLabel *rgbOriginalLabel;
    QLabel *redImageLabel, *greenImageLabel, *blueImageLabel;
    QLabel *redHistLabel, *greenHistLabel, *blueHistLabel;

    // --- Histogram tab – Equalize / Normalize ---
    QLabel *eqOrigLabel, *eqImageLabel, *eqHistLabel, *eqCDFLabel;
    QLabel *normOrigLabel, *normImageLabel, *normHistLabel, *normCDFLabel;
    cv::Mat equalizedResult, normalizedResult;

    // --- Threshold tab ---
    QLabel *threshOriginalLabel;
    QLabel *globalLabel, *localLabel;
    QSlider *globalThreshSlider;
    QSlider *localBlockSlider, *localConstSlider;
    QLabel *globalThreshVal, *localBlockVal, *localConstVal;
    cv::Mat globalResult, localResult;

    // --- Frequency tab ---
    QLabel *freqOriginalLabel;
    QLabel *lowPassLabel, *highPassLabel;
    QSlider *lowPassCutoffSlider, *highPassCutoffSlider;
    QLabel *lowPassCutoffVal, *highPassCutoffVal;
    cv::Mat lowPassResult, highPassResult;

    // --- Hybrid tab ---
    QLabel *hybridImg1Label, *hybridImg2Label, *hybridResultLabel;
    QSlider *hybridCutoff1Slider, *hybridCutoff2Slider;
    QLabel *hybridCutoff1Val, *hybridCutoff2Val;
    QRadioButton *hybridModeFirstLow, *hybridModeFirstHigh;
    cv::Mat hybridResult;

    // --- Noise+Filter tab ---
    QLabel *nfOriginalLabel;
    QLabel *nfNoisyLabel, *nfFilteredLabel;
    QLabel *nfMetricsLabel;
    QComboBox *nfNoiseCombo, *nfFilterCombo;
    QWidget *nfUniformWidget, *nfGaussianWidget, *nfSPWidget;
    QSlider *nfUniformLow, *nfUniformHigh;
    QSlider *nfGaussianMean, *nfGaussianStd;
    QSlider *nfSPProb;
    QLabel *nfUniformLowVal, *nfUniformHighVal;
    QLabel *nfGaussianMeanVal, *nfGaussianStdVal;
    QLabel *nfSPProbVal;
    QWidget *nfAvgWidget, *nfGaussFilterWidget, *nfMedianWidget;
    QSlider *nfAvgKernel, *nfGaussKernel, *nfGaussSigma, *nfMedianKernel;
    QLabel *nfAvgKernelVal, *nfGaussKernelVal, *nfGaussSigmaVal, *nfMedianKernelVal;
    cv::Mat nfNoisyImage, nfFilteredImage;
};

#endif // MAINWINDOW_H