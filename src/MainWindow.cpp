#include "MainWindow.h"
#include "Utils.h"
#include "Noise.h"
#include "Filters.h"
#include "EdgeDetection.h"
#include "Histogram.h"
#include "Threshold.h"
#include "FrequencyDomain.h"
#include "Hybrid.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QScrollArea>
#include <QMessageBox>
#include <QLabel>
#include <QGroupBox>
#include <QTabWidget>
#include <QSlider>
#include <QRadioButton>
#include <QComboBox>
#include <QToolBar>
#include <QMenuBar>
#include <QAction>
#include <QStatusBar>
#include <QDialog>
#include <QScreen>
#include <QGuiApplication>
#include <QApplication>

// Light and dark stylesheets (unchanged)
static const QString lightStyle = R"(
    QWidget { background-color: #f0f0f0; color: #000000; }
    QGroupBox { border: 2px solid #cccccc; border-radius: 5px; margin-top: 1ex; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
    QLabel { background-color: transparent; color: #000000; }
    QLabel[image="true"] { border: 1px solid #aaa; background: transparent; }
    QSlider::groove:horizontal { border: 1px solid #bbb; background: white; height: 6px; border-radius: 3px; }
    QSlider::handle:horizontal { background: #888; width: 18px; margin: -5px 0; border-radius: 9px; }
    QComboBox, QPushButton { background-color: #e0e0e0; border: 1px solid #aaa; padding: 5px; border-radius: 3px; }
    QComboBox:hover, QPushButton:hover { background-color: #d0d0d0; }
    QTabWidget::pane { border: 1px solid #ccc; background: #f0f0f0; }
    QTabBar::tab { background: #e0e0e0; padding: 8px; }
    QTabBar::tab:selected { background: #f0f0f0; }
    QRadioButton { color: #000000; }
    QRadioButton::indicator { width: 13px; height: 13px; }
    QRadioButton::indicator::unchecked { border: 1px solid #888; background: white; }
    QRadioButton::indicator::checked { border: 1px solid #333; background: #ccc; }
)";

static const QString darkStyle = R"(
    QWidget { background-color: #2b2b2b; color: #f0f0f0; }
    QGroupBox { border: 2px solid #555555; border-radius: 5px; margin-top: 1ex; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; color: #f0f0f0; }
    QLabel { background-color: transparent; color: #f0f0f0; }
    QLabel[image="true"] { border: 1px solid #888; background: transparent; }
    QSlider::groove:horizontal { border: 1px solid #555; background: #3c3c3c; height: 6px; border-radius: 3px; }
    QSlider::handle:horizontal { background: #aaa; width: 18px; margin: -5px 0; border-radius: 9px; }
    QComboBox, QPushButton { background-color: #3c3c3c; border: 1px solid #666; padding: 5px; border-radius: 3px; color: #f0f0f0; }
    QComboBox:hover, QPushButton:hover { background-color: #4c4c4c; }
    QComboBox QAbstractItemView { background-color: #3c3c3c; color: #f0f0f0; }
    QTabWidget::pane { border: 1px solid #555; background: #2b2b2b; }
    QTabBar::tab { background: #3c3c3c; color: #f0f0f0; padding: 8px; }
    QTabBar::tab:selected { background: #4c4c4c; }
    QRadioButton { color: #f0f0f0; }
    QRadioButton::indicator { width: 13px; height: 13px; }
    QRadioButton::indicator::unchecked { border: 1px solid #888; background: #3c3c3c; }
    QRadioButton::indicator::checked { border: 1px solid #aaa; background: #5a5a5a; }
)";

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , secondValid(false)
    , fftValid(false)
    , secondFFTValid(false) {
    setupUI();
    createToolBar();
    applyTheme(false);
}

void MainWindow::setupImageLabel(QLabel* label) {
    label->setProperty("image", true);
    label->setMinimumSize(150, 150);
    label->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    label->setAlignment(Qt::AlignCenter);
    label->installEventFilter(this);
}

void MainWindow::showImage(QLabel* label, const cv::Mat& mat) {
    if (mat.empty()) {
        label->clear();
        imageMap.remove(label);
        return;
    }
    QImage qimg = Utils::matToQImage(mat);
    QSize targetSize = label->contentsRect().size();
    if (targetSize.isEmpty() || targetSize.width() < 10 || targetSize.height() < 10)
        targetSize = label->size();
    QPixmap pix = QPixmap::fromImage(qimg).scaled(targetSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    label->setPixmap(pix);
    imageMap[label] = const_cast<cv::Mat*>(&mat);
}

void MainWindow::createToolBar() {
    QToolBar *toolBar = addToolBar("File");
    QAction *loadAction = toolBar->addAction("Load Image");
    connect(loadAction, &QAction::triggered, this, &MainWindow::handleLoad);
    QAction *saveAction = toolBar->addAction("Save Result");
    connect(saveAction, &QAction::triggered, this, &MainWindow::handleSave);
}

void MainWindow::setupUI() {
    auto *central = new QWidget(this);
    auto *mainLayout = new QVBoxLayout(central);

    QTabWidget *tabs = new QTabWidget();
    tabs->setStyleSheet("QTabWidget::pane { border: 1px solid #ccc; background: #f9f9f9; }");

    auto createScrollTab = [&](QWidget *content) -> QWidget* {
        QScrollArea *scroll = new QScrollArea;
        scroll->setWidgetResizable(true);
        scroll->setWidget(content);
        return scroll;
    };

    // --- Noise Tab ---
    QWidget *noiseTabContent = new QWidget();
    QGridLayout *noiseLayout = new QGridLayout(noiseTabContent);
    noiseLayout->setSpacing(15);

    QGroupBox *origGroup = new QGroupBox("Original Image");
    origGroup->setAlignment(Qt::AlignCenter);
    origGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *origLayout = new QVBoxLayout(origGroup);
    noiseOriginalLabel = new QLabel();
    setupImageLabel(noiseOriginalLabel);
    origLayout->addWidget(noiseOriginalLabel, 1); // stretch factor 1
    noiseLayout->addWidget(origGroup, 0, 0, 2, 1);

    // Uniform
    QGroupBox *uniformGroup = new QGroupBox("Uniform Noise");
    uniformGroup->setAlignment(Qt::AlignCenter);
    uniformGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *uniformLayout = new QVBoxLayout(uniformGroup);
    uniformLabel = new QLabel();
    setupImageLabel(uniformLabel);
    uniformLayout->addWidget(uniformLabel, 1);
    uniformLowSlider = new QSlider(Qt::Horizontal); uniformLowSlider->setRange(0,100); uniformLowSlider->setValue(0);
    uniformHighSlider = new QSlider(Qt::Horizontal); uniformHighSlider->setRange(0,100); uniformHighSlider->setValue(50);
    uniformLowVal = new QLabel("0");
    uniformHighVal = new QLabel("50");
    uniformLayout->addWidget(new QLabel("Low:"));
    uniformLayout->addWidget(uniformLowSlider);
    uniformLayout->addWidget(uniformLowVal);
    uniformLayout->addWidget(new QLabel("High:"));
    uniformLayout->addWidget(uniformHighSlider);
    uniformLayout->addWidget(uniformHighVal);
    noiseLayout->addWidget(uniformGroup, 0, 1);

    // Gaussian
    QGroupBox *gaussianGroup = new QGroupBox("Gaussian Noise");
    gaussianGroup->setAlignment(Qt::AlignCenter);
    gaussianGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *gaussianLayout = new QVBoxLayout(gaussianGroup);
    gaussianLabel = new QLabel();
    setupImageLabel(gaussianLabel);
    gaussianLayout->addWidget(gaussianLabel, 1);
    gaussianMeanSlider = new QSlider(Qt::Horizontal); gaussianMeanSlider->setRange(-100,100); gaussianMeanSlider->setValue(0);
    gaussianStdSlider = new QSlider(Qt::Horizontal); gaussianStdSlider->setRange(0,100); gaussianStdSlider->setValue(25);
    gaussianMeanVal = new QLabel("0");
    gaussianStdVal = new QLabel("25");
    gaussianLayout->addWidget(new QLabel("Mean:"));
    gaussianLayout->addWidget(gaussianMeanSlider);
    gaussianLayout->addWidget(gaussianMeanVal);
    gaussianLayout->addWidget(new QLabel("StdDev:"));
    gaussianLayout->addWidget(gaussianStdSlider);
    gaussianLayout->addWidget(gaussianStdVal);
    noiseLayout->addWidget(gaussianGroup, 0, 2);

    // Salt & Pepper
    QGroupBox *spGroup = new QGroupBox("Salt & Pepper");
    spGroup->setAlignment(Qt::AlignCenter);
    spGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *spLayout = new QVBoxLayout(spGroup);
    spLabel = new QLabel();
    setupImageLabel(spLabel);
    spLayout->addWidget(spLabel, 1);
    spProbSlider = new QSlider(Qt::Horizontal); spProbSlider->setRange(1,100); spProbSlider->setValue(5);
    spProbVal = new QLabel("0.05");
    spLayout->addWidget(new QLabel("Probability (%):"));
    spLayout->addWidget(spProbSlider);
    spLayout->addWidget(spProbVal);
    noiseLayout->addWidget(spGroup, 0, 3);

    noiseLayout->setRowStretch(0, 1);
    for (int c = 0; c < 4; ++c) noiseLayout->setColumnStretch(c, 1);
    tabs->addTab(createScrollTab(noiseTabContent), "Noise");

    connect(uniformLowSlider, &QSlider::valueChanged, this, &MainWindow::updateUniformLabels);
    connect(uniformHighSlider, &QSlider::valueChanged, this, &MainWindow::updateUniformLabels);
    connect(uniformLowSlider, &QSlider::sliderReleased, this, &MainWindow::updateUniformNoise);
    connect(uniformHighSlider, &QSlider::sliderReleased, this, &MainWindow::updateUniformNoise);

    connect(gaussianMeanSlider, &QSlider::valueChanged, this, &MainWindow::updateGaussianLabels);
    connect(gaussianStdSlider, &QSlider::valueChanged, this, &MainWindow::updateGaussianLabels);
    connect(gaussianMeanSlider, &QSlider::sliderReleased, this, &MainWindow::updateGaussianNoise);
    connect(gaussianStdSlider, &QSlider::sliderReleased, this, &MainWindow::updateGaussianNoise);

    connect(spProbSlider, &QSlider::valueChanged, this, &MainWindow::updateSPLabels);
    connect(spProbSlider, &QSlider::sliderReleased, this, &MainWindow::updateSaltPepperNoise);

    // --- Filters Tab ---
    QWidget *filterTabContent = new QWidget();
    QGridLayout *filterLayout = new QGridLayout(filterTabContent);
    filterLayout->setSpacing(15);

    origGroup = new QGroupBox("Original Image");
    origGroup->setAlignment(Qt::AlignCenter);
    origGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    origLayout = new QVBoxLayout(origGroup);
    filterOriginalLabel = new QLabel();
    setupImageLabel(filterOriginalLabel);
    origLayout->addWidget(filterOriginalLabel, 1);
    filterLayout->addWidget(origGroup, 0, 0, 2, 1);

    // Average
    QGroupBox *avgGroup = new QGroupBox("Average Filter");
    avgGroup->setAlignment(Qt::AlignCenter);
    avgGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *avgLayout = new QVBoxLayout(avgGroup);
    averageLabel = new QLabel();
    setupImageLabel(averageLabel);
    avgLayout->addWidget(averageLabel, 1);
    avgKernelSlider = new QSlider(Qt::Horizontal); avgKernelSlider->setRange(3,15); avgKernelSlider->setSingleStep(2); avgKernelSlider->setValue(3);
    avgKernelVal = new QLabel("3");
    avgLayout->addWidget(new QLabel("Kernel size:"));
    avgLayout->addWidget(avgKernelSlider);
    avgLayout->addWidget(avgKernelVal);
    filterLayout->addWidget(avgGroup, 0, 1);

    // Gaussian
    QGroupBox *gaussFilterGroup = new QGroupBox("Gaussian Filter");
    gaussFilterGroup->setAlignment(Qt::AlignCenter);
    gaussFilterGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *gaussFilterLayout = new QVBoxLayout(gaussFilterGroup);
    gaussFilterLabel = new QLabel();
    setupImageLabel(gaussFilterLabel);
    gaussFilterLayout->addWidget(gaussFilterLabel, 1);
    gaussKernelSlider = new QSlider(Qt::Horizontal); gaussKernelSlider->setRange(3,15); gaussKernelSlider->setSingleStep(2); gaussKernelSlider->setValue(3);
    gaussSigmaSlider = new QSlider(Qt::Horizontal); gaussSigmaSlider->setRange(1,50); gaussSigmaSlider->setValue(10);
    gaussKernelVal = new QLabel("3");
    gaussSigmaVal = new QLabel("1.0");
    gaussFilterLayout->addWidget(new QLabel("Kernel:"));
    gaussFilterLayout->addWidget(gaussKernelSlider);
    gaussFilterLayout->addWidget(gaussKernelVal);
    gaussFilterLayout->addWidget(new QLabel("Sigma (x10):"));
    gaussFilterLayout->addWidget(gaussSigmaSlider);
    gaussFilterLayout->addWidget(gaussSigmaVal);
    filterLayout->addWidget(gaussFilterGroup, 0, 2);

    // Median
    QGroupBox *medianGroup = new QGroupBox("Median Filter");
    medianGroup->setAlignment(Qt::AlignCenter);
    medianGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *medianLayout = new QVBoxLayout(medianGroup);
    medianLabel = new QLabel();
    setupImageLabel(medianLabel);
    medianLayout->addWidget(medianLabel, 1);
    medianKernelSlider = new QSlider(Qt::Horizontal); medianKernelSlider->setRange(3,15); medianKernelSlider->setSingleStep(2); medianKernelSlider->setValue(3);
    medianKernelVal = new QLabel("3");
    medianLayout->addWidget(new QLabel("Kernel:"));
    medianLayout->addWidget(medianKernelSlider);
    medianLayout->addWidget(medianKernelVal);
    filterLayout->addWidget(medianGroup, 0, 3);

    filterLayout->setRowStretch(0, 1);
    for (int c = 0; c < 4; ++c) filterLayout->setColumnStretch(c, 1);
    tabs->addTab(createScrollTab(filterTabContent), "Filters");

    connect(avgKernelSlider, &QSlider::valueChanged, this, &MainWindow::updateAvgLabels);
    connect(avgKernelSlider, &QSlider::sliderReleased, this, &MainWindow::updateAverageFilter);

    connect(gaussKernelSlider, &QSlider::valueChanged, this, &MainWindow::updateGaussFilterLabels);
    connect(gaussSigmaSlider, &QSlider::valueChanged, this, &MainWindow::updateGaussFilterLabels);
    connect(gaussKernelSlider, &QSlider::sliderReleased, this, &MainWindow::updateGaussianFilter);
    connect(gaussSigmaSlider, &QSlider::sliderReleased, this, &MainWindow::updateGaussianFilter);

    connect(medianKernelSlider, &QSlider::valueChanged, this, &MainWindow::updateMedianLabels);
    connect(medianKernelSlider, &QSlider::sliderReleased, this, &MainWindow::updateMedianFilter);

    // --- Edge Tab ---
    QWidget *edgeTabContent = new QWidget();
    QGridLayout *edgeLayout = new QGridLayout(edgeTabContent);
    edgeLayout->setSpacing(10);

    origGroup = new QGroupBox("Original");
    origGroup->setAlignment(Qt::AlignCenter);
    origGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    origLayout = new QVBoxLayout(origGroup);
    edgeOriginalLabel = new QLabel();
    setupImageLabel(edgeOriginalLabel);
    origLayout->addWidget(edgeOriginalLabel, 1);
    edgeLayout->addWidget(origGroup, 0, 0, 2, 1);

    // Sobel
    QGroupBox *sobelGroup = new QGroupBox("Sobel");
    sobelGroup->setAlignment(Qt::AlignCenter);
    sobelGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *sobelLayout = new QVBoxLayout(sobelGroup);
    sobelLabel = new QLabel();
    setupImageLabel(sobelLabel);
    sobelLayout->addWidget(sobelLabel, 1);
    edgeLayout->addWidget(sobelGroup, 0, 1);

    // Sobel X
    QGroupBox *sobelXGroup = new QGroupBox("Sobel X");
    sobelXGroup->setAlignment(Qt::AlignCenter);
    sobelXGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *sobelXLayout = new QVBoxLayout(sobelXGroup);
    sobelXLabel = new QLabel();
    setupImageLabel(sobelXLabel);
    sobelXLayout->addWidget(sobelXLabel, 1);
    edgeLayout->addWidget(sobelXGroup, 0, 2);

    // Sobel Y
    QGroupBox *sobelYGroup = new QGroupBox("Sobel Y");
    sobelYGroup->setAlignment(Qt::AlignCenter);
    sobelYGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *sobelYLayout = new QVBoxLayout(sobelYGroup);
    sobelYLabel = new QLabel();
    setupImageLabel(sobelYLabel);
    sobelYLayout->addWidget(sobelYLabel, 1);
    edgeLayout->addWidget(sobelYGroup, 0, 3);

    // Roberts
    QGroupBox *robertsGroup = new QGroupBox("Roberts");
    robertsGroup->setAlignment(Qt::AlignCenter);
    robertsGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *robertsLayout = new QVBoxLayout(robertsGroup);
    robertsLabel = new QLabel();
    setupImageLabel(robertsLabel);
    robertsLayout->addWidget(robertsLabel, 1);
    edgeLayout->addWidget(robertsGroup, 1, 1);

    // Prewitt
    QGroupBox *prewittGroup = new QGroupBox("Prewitt");
    prewittGroup->setAlignment(Qt::AlignCenter);
    prewittGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *prewittLayout = new QVBoxLayout(prewittGroup);
    prewittLabel = new QLabel();
    setupImageLabel(prewittLabel);
    prewittLayout->addWidget(prewittLabel, 1);
    edgeLayout->addWidget(prewittGroup, 1, 2);

    // Canny
    QGroupBox *cannyGroup = new QGroupBox("Canny");
    cannyGroup->setAlignment(Qt::AlignCenter);
    cannyGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *cannyLayout = new QVBoxLayout(cannyGroup);
    cannyLabel = new QLabel();
    setupImageLabel(cannyLabel);
    cannyLayout->addWidget(cannyLabel, 1);
    cannyLowSlider = new QSlider(Qt::Horizontal); cannyLowSlider->setRange(1,500); cannyLowSlider->setValue(100);
    cannyHighSlider = new QSlider(Qt::Horizontal); cannyHighSlider->setRange(1,500); cannyHighSlider->setValue(200);
    cannyLowVal = new QLabel("100");
    cannyHighVal = new QLabel("200");
    cannyLayout->addWidget(new QLabel("Low:"));
    cannyLayout->addWidget(cannyLowSlider);
    cannyLayout->addWidget(cannyLowVal);
    cannyLayout->addWidget(new QLabel("High:"));
    cannyLayout->addWidget(cannyHighSlider);
    cannyLayout->addWidget(cannyHighVal);
    edgeLayout->addWidget(cannyGroup, 1, 3);

    edgeLayout->setRowStretch(0, 1);
    edgeLayout->setRowStretch(1, 1);
    for (int c = 0; c < 4; ++c) edgeLayout->setColumnStretch(c, 1);
    tabs->addTab(createScrollTab(edgeTabContent), "Edge");

    connect(cannyLowSlider, &QSlider::valueChanged, this, &MainWindow::updateCannyLabels);
    connect(cannyHighSlider, &QSlider::valueChanged, this, &MainWindow::updateCannyLabels);
    connect(cannyLowSlider, &QSlider::sliderReleased, this, &MainWindow::updateCanny);
    connect(cannyHighSlider, &QSlider::sliderReleased, this, &MainWindow::updateCanny);

    // --- Histogram Tab ---
    QWidget *histTabContent = new QWidget();
    QVBoxLayout *histMainLayout = new QVBoxLayout(histTabContent);
    QTabWidget *histSubTabs = new QTabWidget();

    // RGB Channels sub-tab
    QWidget *rgbTabContent = new QWidget();
    QGridLayout *rgbLayout = new QGridLayout(rgbTabContent);
    rgbLayout->setSpacing(10);

    origGroup = new QGroupBox("Original");
    origGroup->setAlignment(Qt::AlignCenter);
    origGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    origLayout = new QVBoxLayout(origGroup);
    rgbOriginalLabel = new QLabel();
    setupImageLabel(rgbOriginalLabel);
    origLayout->addWidget(rgbOriginalLabel, 1);
    rgbLayout->addWidget(origGroup, 0, 0, 2, 1);

    // Red
    QGroupBox *redGroup = new QGroupBox("Red Channel");
    redGroup->setAlignment(Qt::AlignCenter);
    redGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *redLayout = new QVBoxLayout(redGroup);
    redImageLabel = new QLabel();
    setupImageLabel(redImageLabel);
    redLayout->addWidget(redImageLabel, 1);
    redHistLabel = new QLabel();
    setupImageLabel(redHistLabel);
    redLayout->addWidget(redHistLabel, 1);
    rgbLayout->addWidget(redGroup, 0, 1);

    // Green
    QGroupBox *greenGroup = new QGroupBox("Green Channel");
    greenGroup->setAlignment(Qt::AlignCenter);
    greenGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *greenLayout = new QVBoxLayout(greenGroup);
    greenImageLabel = new QLabel();
    setupImageLabel(greenImageLabel);
    greenLayout->addWidget(greenImageLabel, 1);
    greenHistLabel = new QLabel();
    setupImageLabel(greenHistLabel);
    greenLayout->addWidget(greenHistLabel, 1);
    rgbLayout->addWidget(greenGroup, 0, 2);

    // Blue
    QGroupBox *blueGroup = new QGroupBox("Blue Channel");
    blueGroup->setAlignment(Qt::AlignCenter);
    blueGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *blueLayout = new QVBoxLayout(blueGroup);
    blueImageLabel = new QLabel();
    setupImageLabel(blueImageLabel);
    blueLayout->addWidget(blueImageLabel, 1);
    blueHistLabel = new QLabel();
    setupImageLabel(blueHistLabel);
    blueLayout->addWidget(blueHistLabel, 1);
    rgbLayout->addWidget(blueGroup, 0, 3);

    QPushButton *updateRGB = new QPushButton("Update RGB Channels");
    rgbLayout->addWidget(updateRGB, 1, 0, 1, 4);
    connect(updateRGB, &QPushButton::clicked, this, &MainWindow::updateRGBChannels);

    rgbLayout->setRowStretch(0, 1);
    for (int c = 0; c < 4; ++c) rgbLayout->setColumnStretch(c, 1);
    histSubTabs->addTab(createScrollTab(rgbTabContent), "RGB Channels");

    // Equalize/Normalize sub-tab
    QWidget *eqNormTabContent = new QWidget();
    QGridLayout *eqNormLayout = new QGridLayout(eqNormTabContent);
    eqNormLayout->setSpacing(10);

    origGroup = new QGroupBox("Original");
    origGroup->setAlignment(Qt::AlignCenter);
    origGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    origLayout = new QVBoxLayout(origGroup);
    eqOrigLabel = new QLabel();
    setupImageLabel(eqOrigLabel);
    origLayout->addWidget(eqOrigLabel, 1);
    eqNormLayout->addWidget(origGroup, 0, 0, 3, 1);

    QGroupBox *eqGroup = new QGroupBox("Equalize");
    eqGroup->setAlignment(Qt::AlignCenter);
    eqGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *eqLayout = new QVBoxLayout(eqGroup);
    eqImageLabel = new QLabel();
    setupImageLabel(eqImageLabel);
    eqLayout->addWidget(eqImageLabel, 1);
    eqHistLabel = new QLabel();
    setupImageLabel(eqHistLabel);
    eqLayout->addWidget(eqHistLabel, 1);
    eqCDFLabel = new QLabel();
    setupImageLabel(eqCDFLabel);
    eqLayout->addWidget(eqCDFLabel, 1);
    QPushButton *btnEq = new QPushButton("Update");
    eqLayout->addWidget(btnEq);
    eqNormLayout->addWidget(eqGroup, 0, 1, 3, 1);
    connect(btnEq, &QPushButton::clicked, this, &MainWindow::updateEqualize);

    QGroupBox *normGroup = new QGroupBox("Normalize");
    normGroup->setAlignment(Qt::AlignCenter);
    normGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *normLayout = new QVBoxLayout(normGroup);
    normImageLabel = new QLabel();
    setupImageLabel(normImageLabel);
    normLayout->addWidget(normImageLabel, 1);
    normHistLabel = new QLabel();
    setupImageLabel(normHistLabel);
    normLayout->addWidget(normHistLabel, 1);
    normCDFLabel = new QLabel();
    setupImageLabel(normCDFLabel);
    normLayout->addWidget(normCDFLabel, 1);
    QPushButton *btnNorm = new QPushButton("Update");
    normLayout->addWidget(btnNorm);
    eqNormLayout->addWidget(normGroup, 0, 2, 3, 1);
    connect(btnNorm, &QPushButton::clicked, this, &MainWindow::updateNormalize);

    eqNormLayout->setRowStretch(0, 1);
    eqNormLayout->setColumnStretch(0, 1);
    eqNormLayout->setColumnStretch(1, 1);
    eqNormLayout->setColumnStretch(2, 1);
    histSubTabs->addTab(createScrollTab(eqNormTabContent), "Equalize/Normalize");

    histMainLayout->addWidget(histSubTabs);
    tabs->addTab(createScrollTab(histTabContent), "Histogram");

    // --- Threshold Tab ---
    QWidget *threshTabContent = new QWidget();
    QGridLayout *threshLayout = new QGridLayout(threshTabContent);
    threshLayout->setSpacing(15);

    origGroup = new QGroupBox("Original");
    origGroup->setAlignment(Qt::AlignCenter);
    origGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    origLayout = new QVBoxLayout(origGroup);
    threshOriginalLabel = new QLabel();
    setupImageLabel(threshOriginalLabel);
    origLayout->addWidget(threshOriginalLabel, 1);
    threshLayout->addWidget(origGroup, 0, 0, 2, 1);

    QGroupBox *globalGroup = new QGroupBox("Global Threshold");
    globalGroup->setAlignment(Qt::AlignCenter);
    globalGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *globalLayout = new QVBoxLayout(globalGroup);
    globalLabel = new QLabel();
    setupImageLabel(globalLabel);
    globalLayout->addWidget(globalLabel, 1);
    globalThreshSlider = new QSlider(Qt::Horizontal); globalThreshSlider->setRange(0,255); globalThreshSlider->setValue(128);
    globalThreshVal = new QLabel("128");
    globalLayout->addWidget(new QLabel("Threshold:"));
    globalLayout->addWidget(globalThreshSlider);
    globalLayout->addWidget(globalThreshVal);
    threshLayout->addWidget(globalGroup, 0, 1);

    QGroupBox *localGroup = new QGroupBox("Local Threshold");
    localGroup->setAlignment(Qt::AlignCenter);
    localGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *localLayout = new QVBoxLayout(localGroup);
    localLabel = new QLabel();
    setupImageLabel(localLabel);
    localLayout->addWidget(localLabel, 1);
    localBlockSlider = new QSlider(Qt::Horizontal); localBlockSlider->setRange(3,51); localBlockSlider->setSingleStep(2); localBlockSlider->setValue(11);
    localConstSlider = new QSlider(Qt::Horizontal); localConstSlider->setRange(0,20); localConstSlider->setValue(2);
    localBlockVal = new QLabel("11");
    localConstVal = new QLabel("2");
    localLayout->addWidget(new QLabel("Block size:"));
    localLayout->addWidget(localBlockSlider);
    localLayout->addWidget(localBlockVal);
    localLayout->addWidget(new QLabel("Constant:"));
    localLayout->addWidget(localConstSlider);
    localLayout->addWidget(localConstVal);
    threshLayout->addWidget(localGroup, 0, 2);

    threshLayout->setRowStretch(0, 1);
    threshLayout->setColumnStretch(0, 1);
    threshLayout->setColumnStretch(1, 1);
    threshLayout->setColumnStretch(2, 1);
    tabs->addTab(createScrollTab(threshTabContent), "Threshold");

    connect(globalThreshSlider, &QSlider::valueChanged, this, &MainWindow::updateGlobalLabels);
    connect(globalThreshSlider, &QSlider::sliderReleased, this, &MainWindow::updateGlobalThreshold);

    connect(localBlockSlider, &QSlider::valueChanged, this, &MainWindow::updateLocalLabels);
    connect(localConstSlider, &QSlider::valueChanged, this, &MainWindow::updateLocalLabels);
    connect(localBlockSlider, &QSlider::sliderReleased, this, &MainWindow::updateLocalThreshold);
    connect(localConstSlider, &QSlider::sliderReleased, this, &MainWindow::updateLocalThreshold);

    // --- Frequency Tab ---
    QWidget *freqTabContent = new QWidget();
    QGridLayout *freqLayout = new QGridLayout(freqTabContent);
    freqLayout->setSpacing(15);

    origGroup = new QGroupBox("Original");
    origGroup->setAlignment(Qt::AlignCenter);
    origGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    origLayout = new QVBoxLayout(origGroup);
    freqOriginalLabel = new QLabel();
    setupImageLabel(freqOriginalLabel);
    origLayout->addWidget(freqOriginalLabel, 1);
    freqLayout->addWidget(origGroup, 0, 0, 2, 1);

    QGroupBox *lpGroup = new QGroupBox("Low Pass");
    lpGroup->setAlignment(Qt::AlignCenter);
    lpGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *lpLayout = new QVBoxLayout(lpGroup);
    lowPassLabel = new QLabel();
    setupImageLabel(lowPassLabel);
    lpLayout->addWidget(lowPassLabel, 1);
    lowPassCutoffSlider = new QSlider(Qt::Horizontal); lowPassCutoffSlider->setRange(1,100); lowPassCutoffSlider->setValue(30);
    lowPassCutoffVal = new QLabel("30");
    lpLayout->addWidget(new QLabel("Cutoff:"));
    lpLayout->addWidget(lowPassCutoffSlider);
    lpLayout->addWidget(lowPassCutoffVal);
    freqLayout->addWidget(lpGroup, 0, 1);

    QGroupBox *hpGroup = new QGroupBox("High Pass");
    hpGroup->setAlignment(Qt::AlignCenter);
    hpGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *hpLayout = new QVBoxLayout(hpGroup);
    highPassLabel = new QLabel();
    setupImageLabel(highPassLabel);
    hpLayout->addWidget(highPassLabel, 1);
    highPassCutoffSlider = new QSlider(Qt::Horizontal); highPassCutoffSlider->setRange(1,100); highPassCutoffSlider->setValue(30);
    highPassCutoffVal = new QLabel("30");
    hpLayout->addWidget(new QLabel("Cutoff:"));
    hpLayout->addWidget(highPassCutoffSlider);
    hpLayout->addWidget(highPassCutoffVal);
    freqLayout->addWidget(hpGroup, 0, 2);

    freqLayout->setRowStretch(0, 1);
    freqLayout->setColumnStretch(0, 1);
    freqLayout->setColumnStretch(1, 1);
    freqLayout->setColumnStretch(2, 1);
    tabs->addTab(createScrollTab(freqTabContent), "Frequency");

    connect(lowPassCutoffSlider, &QSlider::valueChanged, this, &MainWindow::updateLowPassLabels);
    connect(lowPassCutoffSlider, &QSlider::sliderReleased, this, &MainWindow::updateLowPass);

    connect(highPassCutoffSlider, &QSlider::valueChanged, this, &MainWindow::updateHighPassLabels);
    connect(highPassCutoffSlider, &QSlider::sliderReleased, this, &MainWindow::updateHighPass);

    // --- Hybrid Tab ---
    QWidget *hybridTabContent = new QWidget();
    QGridLayout *hybridLayout = new QGridLayout(hybridTabContent);
    hybridLayout->setSpacing(15);

    QGroupBox *img1Group = new QGroupBox("Image 1");
    img1Group->setAlignment(Qt::AlignCenter);
    img1Group->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *img1Layout = new QVBoxLayout(img1Group);
    hybridImg1Label = new QLabel();
    setupImageLabel(hybridImg1Label);
    img1Layout->addWidget(hybridImg1Label, 1);
    hybridLayout->addWidget(img1Group, 0, 0);

    QGroupBox *img2Group = new QGroupBox("Image 2");
    img2Group->setAlignment(Qt::AlignCenter);
    img2Group->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *img2Layout = new QVBoxLayout(img2Group);
    hybridImg2Label = new QLabel();
    setupImageLabel(hybridImg2Label);
    img2Layout->addWidget(hybridImg2Label, 1);
    hybridLayout->addWidget(img2Group, 0, 1);

    QGroupBox *resultGroup = new QGroupBox("Hybrid");
    resultGroup->setAlignment(Qt::AlignCenter);
    resultGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *resultLayout = new QVBoxLayout(resultGroup);
    hybridResultLabel = new QLabel();
    setupImageLabel(hybridResultLabel);
    resultLayout->addWidget(hybridResultLabel, 1);
    hybridLayout->addWidget(resultGroup, 0, 2);

    QGroupBox *ctrlGroup = new QGroupBox("Parameters");
    ctrlGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    QVBoxLayout *ctrlLayout = new QVBoxLayout(ctrlGroup);
    QPushButton *btnLoadSecond = new QPushButton("Load Second Image");
    ctrlLayout->addWidget(btnLoadSecond);
    hybridCutoff1Slider = new QSlider(Qt::Horizontal); hybridCutoff1Slider->setRange(1,100); hybridCutoff1Slider->setValue(30);
    hybridCutoff2Slider = new QSlider(Qt::Horizontal); hybridCutoff2Slider->setRange(1,100); hybridCutoff2Slider->setValue(30);
    hybridCutoff1Val = new QLabel("30");
    hybridCutoff2Val = new QLabel("30");
    ctrlLayout->addWidget(new QLabel("Cutoff 1:"));
    ctrlLayout->addWidget(hybridCutoff1Slider);
    ctrlLayout->addWidget(hybridCutoff1Val);
    ctrlLayout->addWidget(new QLabel("Cutoff 2:"));
    ctrlLayout->addWidget(hybridCutoff2Slider);
    ctrlLayout->addWidget(hybridCutoff2Val);
    hybridModeFirstLow = new QRadioButton("Img1 Low / Img2 High");
    hybridModeFirstHigh = new QRadioButton("Img1 High / Img2 Low");
    hybridModeFirstLow->setChecked(true);
    ctrlLayout->addWidget(hybridModeFirstLow);
    ctrlLayout->addWidget(hybridModeFirstHigh);
    hybridLayout->addWidget(ctrlGroup, 1, 0, 1, 3);

    hybridLayout->setRowStretch(0, 1);
    hybridLayout->setRowStretch(1, 0);
    hybridLayout->setColumnStretch(0, 1);
    hybridLayout->setColumnStretch(1, 1);
    hybridLayout->setColumnStretch(2, 1);
    tabs->addTab(createScrollTab(hybridTabContent), "Hybrid");

    connect(btnLoadSecond, &QPushButton::clicked, this, &MainWindow::loadSecondImage);
    connect(hybridCutoff1Slider, &QSlider::valueChanged, this, &MainWindow::updateHybridLabels);
    connect(hybridCutoff2Slider, &QSlider::valueChanged, this, &MainWindow::updateHybridLabels);
    connect(hybridCutoff1Slider, &QSlider::sliderReleased, this, &MainWindow::updateHybrid);
    connect(hybridCutoff2Slider, &QSlider::sliderReleased, this, &MainWindow::updateHybrid);
    connect(hybridModeFirstLow, &QRadioButton::toggled, this, &MainWindow::updateHybrid);
    connect(hybridModeFirstHigh, &QRadioButton::toggled, this, &MainWindow::updateHybrid);

    // --- Noise+Filter Tab ---
    QWidget *nfTabContent = new QWidget();
    QGridLayout *nfLayout = new QGridLayout(nfTabContent);
    nfLayout->setSpacing(15);

    origGroup = new QGroupBox("Original");
    origGroup->setAlignment(Qt::AlignCenter);
    origGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    origLayout = new QVBoxLayout(origGroup);
    nfOriginalLabel = new QLabel();
    setupImageLabel(nfOriginalLabel);
    origLayout->addWidget(nfOriginalLabel, 1);
    nfLayout->addWidget(origGroup, 0, 0, 2, 1);

    QGroupBox *noisyGroup = new QGroupBox("Noisy Image");
    noisyGroup->setAlignment(Qt::AlignCenter);
    noisyGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *noisyLayout = new QVBoxLayout(noisyGroup);
    nfNoisyLabel = new QLabel();
    setupImageLabel(nfNoisyLabel);
    noisyLayout->addWidget(nfNoisyLabel, 1);
    nfLayout->addWidget(noisyGroup, 0, 1);

    QGroupBox *filteredGroup = new QGroupBox("Filtered Image");
    filteredGroup->setAlignment(Qt::AlignCenter);
    filteredGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *filteredLayout = new QVBoxLayout(filteredGroup);
    nfFilteredLabel = new QLabel();
    setupImageLabel(nfFilteredLabel);
    filteredLayout->addWidget(nfFilteredLabel, 1);
    nfLayout->addWidget(filteredGroup, 0, 2);

    QGroupBox *metricsGroup = new QGroupBox("Metrics (Original vs Filtered)");
    metricsGroup->setAlignment(Qt::AlignCenter);
    metricsGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    QVBoxLayout *metricsLayout = new QVBoxLayout(metricsGroup);
    nfMetricsLabel = new QLabel("MSE: --\nPSNR: --\nSNR: --");
    nfMetricsLabel->setAlignment(Qt::AlignCenter);
    metricsLayout->addWidget(nfMetricsLabel);
    nfLayout->addWidget(metricsGroup, 1, 1, 1, 2);

    QGroupBox *noiseCtrlGroup = new QGroupBox("Noise Type");
    noiseCtrlGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    QVBoxLayout *noiseCtrlLayout = new QVBoxLayout(noiseCtrlGroup);
    nfNoiseCombo = new QComboBox();
    nfNoiseCombo->addItems({"Uniform", "Gaussian", "Salt & Pepper"});
    noiseCtrlLayout->addWidget(nfNoiseCombo);

    nfUniformWidget = new QWidget();
    QVBoxLayout *uniformCtrlLayout = new QVBoxLayout(nfUniformWidget);
    nfUniformLow = new QSlider(Qt::Horizontal); nfUniformLow->setRange(0,100); nfUniformLow->setValue(0);
    nfUniformHigh = new QSlider(Qt::Horizontal); nfUniformHigh->setRange(0,100); nfUniformHigh->setValue(50);
    nfUniformLowVal = new QLabel("0");
    nfUniformHighVal = new QLabel("50");
    uniformCtrlLayout->addWidget(new QLabel("Low:"));
    uniformCtrlLayout->addWidget(nfUniformLow);
    uniformCtrlLayout->addWidget(nfUniformLowVal);
    uniformCtrlLayout->addWidget(new QLabel("High:"));
    uniformCtrlLayout->addWidget(nfUniformHigh);
    uniformCtrlLayout->addWidget(nfUniformHighVal);

    nfGaussianWidget = new QWidget();
    QVBoxLayout *gaussianCtrlLayout = new QVBoxLayout(nfGaussianWidget);
    nfGaussianMean = new QSlider(Qt::Horizontal); nfGaussianMean->setRange(-100,100); nfGaussianMean->setValue(0);
    nfGaussianStd = new QSlider(Qt::Horizontal); nfGaussianStd->setRange(0,100); nfGaussianStd->setValue(25);
    nfGaussianMeanVal = new QLabel("0");
    nfGaussianStdVal = new QLabel("25");
    gaussianCtrlLayout->addWidget(new QLabel("Mean:"));
    gaussianCtrlLayout->addWidget(nfGaussianMean);
    gaussianCtrlLayout->addWidget(nfGaussianMeanVal);
    gaussianCtrlLayout->addWidget(new QLabel("StdDev:"));
    gaussianCtrlLayout->addWidget(nfGaussianStd);
    gaussianCtrlLayout->addWidget(nfGaussianStdVal);

    nfSPWidget = new QWidget();
    QVBoxLayout *spCtrlLayout = new QVBoxLayout(nfSPWidget);
    nfSPProb = new QSlider(Qt::Horizontal); nfSPProb->setRange(1,100); nfSPProb->setValue(5);
    nfSPProbVal = new QLabel("0.05");
    spCtrlLayout->addWidget(new QLabel("Probability (%):"));
    spCtrlLayout->addWidget(nfSPProb);
    spCtrlLayout->addWidget(nfSPProbVal);

    noiseCtrlLayout->addWidget(nfUniformWidget);
    noiseCtrlLayout->addWidget(nfGaussianWidget);
    noiseCtrlLayout->addWidget(nfSPWidget);
    nfGaussianWidget->hide();
    nfSPWidget->hide();

    nfLayout->addWidget(noiseCtrlGroup, 2, 0);

    QGroupBox *filterCtrlGroup = new QGroupBox("Filter Type");
    filterCtrlGroup->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    QVBoxLayout *filterCtrlLayout = new QVBoxLayout(filterCtrlGroup);
    nfFilterCombo = new QComboBox();
    nfFilterCombo->addItems({"Average", "Gaussian", "Median"});
    filterCtrlLayout->addWidget(nfFilterCombo);

    nfAvgWidget = new QWidget();
    QVBoxLayout *avgCtrlLayout = new QVBoxLayout(nfAvgWidget);
    nfAvgKernel = new QSlider(Qt::Horizontal); nfAvgKernel->setRange(3,15); nfAvgKernel->setSingleStep(2); nfAvgKernel->setValue(3);
    nfAvgKernelVal = new QLabel("3");
    avgCtrlLayout->addWidget(new QLabel("Kernel:"));
    avgCtrlLayout->addWidget(nfAvgKernel);
    avgCtrlLayout->addWidget(nfAvgKernelVal);

    nfGaussFilterWidget = new QWidget();
    QVBoxLayout *gaussFilterCtrlLayout = new QVBoxLayout(nfGaussFilterWidget);
    nfGaussKernel = new QSlider(Qt::Horizontal); nfGaussKernel->setRange(3,15); nfGaussKernel->setSingleStep(2); nfGaussKernel->setValue(3);
    nfGaussSigma = new QSlider(Qt::Horizontal); nfGaussSigma->setRange(1,50); nfGaussSigma->setValue(10);
    nfGaussKernelVal = new QLabel("3");
    nfGaussSigmaVal = new QLabel("1.0");
    gaussFilterCtrlLayout->addWidget(new QLabel("Kernel:"));
    gaussFilterCtrlLayout->addWidget(nfGaussKernel);
    gaussFilterCtrlLayout->addWidget(nfGaussKernelVal);
    gaussFilterCtrlLayout->addWidget(new QLabel("Sigma (x10):"));
    gaussFilterCtrlLayout->addWidget(nfGaussSigma);
    gaussFilterCtrlLayout->addWidget(nfGaussSigmaVal);

    nfMedianWidget = new QWidget();
    QVBoxLayout *medianCtrlLayout = new QVBoxLayout(nfMedianWidget);
    nfMedianKernel = new QSlider(Qt::Horizontal); nfMedianKernel->setRange(3,15); nfMedianKernel->setSingleStep(2); nfMedianKernel->setValue(3);
    nfMedianKernelVal = new QLabel("3");
    medianCtrlLayout->addWidget(new QLabel("Kernel:"));
    medianCtrlLayout->addWidget(nfMedianKernel);
    medianCtrlLayout->addWidget(nfMedianKernelVal);

    filterCtrlLayout->addWidget(nfAvgWidget);
    filterCtrlLayout->addWidget(nfGaussFilterWidget);
    filterCtrlLayout->addWidget(nfMedianWidget);
    nfGaussFilterWidget->hide();
    nfMedianWidget->hide();

    nfLayout->addWidget(filterCtrlGroup, 2, 1);

    nfLayout->setRowStretch(0, 1);
    nfLayout->setRowStretch(1, 0);
    nfLayout->setRowStretch(2, 0);
    nfLayout->setColumnStretch(0, 1);
    nfLayout->setColumnStretch(1, 1);
    nfLayout->setColumnStretch(2, 1);
    tabs->addTab(createScrollTab(nfTabContent), "Noise+Filter");

    connect(nfNoiseCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onNoiseTypeChanged);
    connect(nfFilterCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onFilterTypeChanged);

    connect(nfUniformLow, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfUniformHigh, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfUniformLow, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);
    connect(nfUniformHigh, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);

    connect(nfGaussianMean, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfGaussianStd, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfGaussianMean, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);
    connect(nfGaussianStd, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);

    connect(nfSPProb, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfSPProb, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);

    connect(nfAvgKernel, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfAvgKernel, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);

    connect(nfGaussKernel, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfGaussSigma, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfGaussKernel, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);
    connect(nfGaussSigma, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);

    connect(nfMedianKernel, &QSlider::valueChanged, this, &MainWindow::updateNoiseFilterLabels);
    connect(nfMedianKernel, &QSlider::sliderReleased, this, &MainWindow::updateNoiseFilter);

    mainLayout->addWidget(tabs);
    setCentralWidget(central);

    QMenuBar *menuBar = new QMenuBar(this);
    QMenu *viewMenu = menuBar->addMenu("View");
    QAction *darkModeAction = viewMenu->addAction("Dark Mode");
    darkModeAction->setCheckable(true);
    connect(darkModeAction, &QAction::toggled, this, &MainWindow::toggleDarkMode);
    setMenuBar(menuBar);

    resize(1300, 800);
}

void MainWindow::applyTheme(bool dark) {
    qApp->setStyleSheet(dark ? darkStyle : lightStyle);
}

void MainWindow::toggleDarkMode(bool checked) {
    applyTheme(checked);
}

void MainWindow::computeMetrics(const cv::Mat& original, const cv::Mat& processed,
                                double& mse, double& psnr, double& snr) {
    cv::Mat origGray, procGray;
    if (original.channels() == 3)
        cv::cvtColor(original, origGray, cv::COLOR_BGR2GRAY);
    else
        origGray = original.clone();
    if (processed.channels() == 3)
        cv::cvtColor(processed, procGray, cv::COLOR_BGR2GRAY);
    else
        procGray = processed.clone();

    cv::Mat diff;
    cv::absdiff(origGray, procGray, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    mse = cv::mean(diff)[0];
    if (mse <= 1e-10) mse = 1e-10;
    psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cv::Scalar meanOrig = cv::mean(origGray);
    double signalPower = meanOrig[0] * meanOrig[0];
    snr = 10.0 * log10(signalPower / mse);
}

void MainWindow::handleLoad() {
    QString path = QFileDialog::getOpenFileName(this, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (path.isEmpty()) return;
    originalFull = cv::imread(path.toStdString());
    if (originalFull.empty()) {
        QMessageBox::critical(this, "Error", "Could not load image.");
        return;
    }
    originalResized = Utils::resizeAspect(originalFull, 512);

    showImage(noiseOriginalLabel, originalResized);
    showImage(filterOriginalLabel, originalResized);
    showImage(edgeOriginalLabel, originalResized);
    showImage(rgbOriginalLabel, originalResized);
    showImage(eqOrigLabel, originalResized);
    showImage(threshOriginalLabel, originalResized);
    showImage(freqOriginalLabel, originalResized);
    showImage(hybridImg1Label, originalResized);
    showImage(nfOriginalLabel, originalResized);

    fftValid = false;
    secondValid = false;
    secondResized = cv::Mat();
    showImage(hybridImg2Label, cv::Mat());

    updateUniformNoise();
    updateGaussianNoise();
    updateSaltPepperNoise();
    updateAverageFilter();
    updateGaussianFilter();
    updateMedianFilter();
    updateSobel();
    updateSobelX();
    updateSobelY();
    updateRoberts();
    updatePrewitt();
    updateCanny();
    updateRGBChannels();
    updateEqualize();
    updateNormalize();
    updateGlobalThreshold();
    updateLocalThreshold();
    updateLowPass();
    updateHighPass();
    updateNoiseFilter();

    updateUniformLabels();
    updateGaussianLabels();
    updateSPLabels();
    updateAvgLabels();
    updateGaussFilterLabels();
    updateMedianLabels();
    updateCannyLabels();
    updateGlobalLabels();
    updateLocalLabels();
    updateLowPassLabels();
    updateHighPassLabels();
    updateHybridLabels();
    updateNoiseFilterLabels();
}

void MainWindow::handleSave() {
    if (!nfFilteredImage.empty()) {
        QString path = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG (*.png);;JPG (*.jpg)");
        if (!path.isEmpty())
            cv::imwrite(path.toStdString(), nfFilteredImage);
    } else {
        QMessageBox::information(this, "Info", "No filtered image to save. Process an image first.");
    }
}

bool MainWindow::eventFilter(QObject *obj, QEvent *event) {
    if (event->type() == QEvent::MouseButtonDblClick) {
        QLabel *label = qobject_cast<QLabel*>(obj);
        if (label && imageMap.contains(label)) {
            cv::Mat *mat = imageMap[label];
            if (mat && !mat->empty()) {
                showEnlargedImage(*mat);
                return true;
            }
        }
    }
    // Handle resize events to update image scaling
    else if (event->type() == QEvent::Resize) {
        QLabel *label = qobject_cast<QLabel*>(obj);
        if (label && imageMap.contains(label)) {
            cv::Mat *mat = imageMap[label];
            if (mat && !mat->empty()) {
                // Re-show the image with the new label size
                showImage(label, *mat);
            }
        }
        // Do not consume the event; let it propagate
        return false;
    }
    return QMainWindow::eventFilter(obj, event);
}

void MainWindow::showEnlargedImage(const cv::Mat& img) {
    QDialog dialog(this);
    dialog.setWindowTitle("Enlarged Image");
    QVBoxLayout *layout = new QVBoxLayout(&dialog);
    QLabel *label = new QLabel();
    label->setAlignment(Qt::AlignCenter);

    QScreen *screen = QGuiApplication::primaryScreen();
    QRect screenGeometry = screen->availableGeometry();
    int maxW = screenGeometry.width() * 0.9;
    int maxH = screenGeometry.height() * 0.9;

    QImage qimg = Utils::matToQImage(img);
    QPixmap pix = QPixmap::fromImage(qimg).scaled(maxW, maxH, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    label->setPixmap(pix);
    layout->addWidget(label);
    dialog.resize(pix.size());
    dialog.exec();
}