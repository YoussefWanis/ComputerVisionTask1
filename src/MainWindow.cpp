#include "MainWindow.h"
#include "ImageProcessor.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QScrollArea>
#include <QMessageBox>
#include <QLabel>
#include <QButtonGroup>
#include <QRadioButton>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), fftValid(false), secondFFTValid(false), hybridMode(true) {
    setupUI();
}

void MainWindow::setupUI() {
    auto *central = new QWidget(this);
    auto *mainLayout = new QHBoxLayout(central);

    tabs = new QTabWidget();
    tabs->setTabPosition(QTabWidget::West);
    tabs->setFixedWidth(320);

    // File tab
    tabs->addTab(createTab("File", {"Open Image", "Save Result", "Reset"}), "File");
    // Noise tab
    tabs->addTab(createTab("Noise", {"Uniform", "Gaussian", "Salt & Pepper", "Show All"}), "Noise");
    // Filters tab
    tabs->addTab(createTab("Filters", {"Average", "Gaussian", "Median", "Show All"}), "Filters");
    // Edge tab
    tabs->addTab(createTab("Edge", {"Sobel", "Roberts", "Prewitt", "Canny", "Show All"}), "Edge");
    // Histogram tab
    tabs->addTab(createTab("Histogram", {"Gray Hist", "RGB Hist", "Equalize", "Normalize"}), "Hist");
    // Threshold tab
    tabs->addTab(createTab("Threshold", {"Global", "Local"}), "Thresh");
    // Frequency tab
    tabs->addTab(createTab("Frequency", {"Low Pass", "High Pass"}), "Freq");
    // Hybrid tab
    tabs->addTab(createTab("Hybrid", {"Load Second", "Create Hybrid"}), "Hybrid");

    // Display area
    auto *scrollArea = new QScrollArea();
    displayContainer = new QWidget();
    displayLayout = new QHBoxLayout(displayContainer);
    displayLayout->setAlignment(Qt::AlignCenter);
    scrollArea->setWidget(displayContainer);
    scrollArea->setWidgetResizable(true);

    mainLayout->addWidget(tabs);
    mainLayout->addWidget(scrollArea);
    setCentralWidget(central);
    resize(1200, 800);
}

QWidget* MainWindow::createTab(const QString& title, const QStringList& actions) {
    auto *w = new QWidget();
    auto *l = new QVBoxLayout(w);
    l->addWidget(new QLabel("<h3>" + title + "</h3>"));

    for (const auto& act : actions) {
        auto *btn = new QPushButton(act);
        btn->setMinimumHeight(40);
        l->addWidget(btn);

        if (act == "Open Image") connect(btn, &QPushButton::clicked, this, &MainWindow::handleLoad);
        else if (act == "Save Result") connect(btn, &QPushButton::clicked, this, &MainWindow::handleSave);
        else if (act == "Reset") connect(btn, &QPushButton::clicked, this, &MainWindow::handleReset);
        else if (act == "Uniform") connect(btn, &QPushButton::clicked, this, &MainWindow::handleUniformNoise);
        else if (act == "Gaussian") connect(btn, &QPushButton::clicked, this, &MainWindow::handleGaussianNoise);
        else if (act == "Salt & Pepper") connect(btn, &QPushButton::clicked, this, &MainWindow::handleSaltPepperNoise);
        else if (act == "Show All" && title == "Noise") connect(btn, &QPushButton::clicked, this, &MainWindow::handleAllNoise);
        else if (act == "Average") connect(btn, &QPushButton::clicked, this, &MainWindow::handleAverageFilter);
        else if (act == "Gaussian" && title == "Filters") connect(btn, &QPushButton::clicked, this, &MainWindow::handleGaussianFilter);
        else if (act == "Median") connect(btn, &QPushButton::clicked, this, &MainWindow::handleMedianFilter);
        else if (act == "Show All" && title == "Filters") connect(btn, &QPushButton::clicked, this, &MainWindow::handleAllFilters);
        else if (act == "Sobel") connect(btn, &QPushButton::clicked, this, &MainWindow::handleSobel);
        else if (act == "Roberts") connect(btn, &QPushButton::clicked, this, &MainWindow::handleRoberts);
        else if (act == "Prewitt") connect(btn, &QPushButton::clicked, this, &MainWindow::handlePrewitt);
        else if (act == "Canny") connect(btn, &QPushButton::clicked, this, &MainWindow::handleCanny);
        else if (act == "Show All" && title == "Edge") connect(btn, &QPushButton::clicked, this, &MainWindow::handleAllEdges);
        else if (act == "Gray Hist") connect(btn, &QPushButton::clicked, this, &MainWindow::handleShowHistogramGray);
        else if (act == "RGB Hist") connect(btn, &QPushButton::clicked, this, &MainWindow::handleShowHistogramRGB);
        else if (act == "Equalize") connect(btn, &QPushButton::clicked, this, &MainWindow::handleEqualize);
        else if (act == "Normalize") connect(btn, &QPushButton::clicked, this, &MainWindow::handleNormalize);
        else if (act == "Global") connect(btn, &QPushButton::clicked, this, &MainWindow::handleGlobalThreshold);
        else if (act == "Local") connect(btn, &QPushButton::clicked, this, &MainWindow::handleLocalThreshold);
        else if (act == "Low Pass") connect(btn, &QPushButton::clicked, this, &MainWindow::handleLowPass);
        else if (act == "High Pass") connect(btn, &QPushButton::clicked, this, &MainWindow::handleHighPass);
        else if (act == "Load Second") connect(btn, &QPushButton::clicked, this, &MainWindow::loadSecondImage);
        else if (act == "Create Hybrid") connect(btn, &QPushButton::clicked, this, &MainWindow::handleHybrid);
    }

    // If this is the Hybrid tab, add radio buttons for mode selection
    if (title == "Hybrid") {
        l->addWidget(new QLabel("Hybrid Mode:"));
        radioFirstLow = new QRadioButton("Image1 Low / Image2 High");
        radioFirstHigh = new QRadioButton("Image1 High / Image2 Low");
        radioFirstLow->setChecked(true); // default
        l->addWidget(radioFirstLow);
        l->addWidget(radioFirstHigh);
        connect(radioFirstLow, &QRadioButton::toggled, this, [this](bool checked) {
            if (checked) hybridMode = true;
        });
        connect(radioFirstHigh, &QRadioButton::toggled, this, [this](bool checked) {
            if (checked) hybridMode = false;
        });
    }

    l->addStretch();
    return w;
}

// -------------------- Display Helpers --------------------
QPixmap MainWindow::scaleAndConvert(const cv::Mat& mat, int maxWidth, int maxHeight) {
    QImage qimg = Utils::matToQImage(mat);
    if (qimg.isNull()) return QPixmap();
    QPixmap pix = QPixmap::fromImage(qimg);
    if (pix.width() > maxWidth || pix.height() > maxHeight) {
        pix = pix.scaled(maxWidth, maxHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }
    return pix;
}

void MainWindow::updateDisplay(const cv::Mat& mat) {
    // Clear layout
    QLayoutItem *child;
    while ((child = displayLayout->takeAt(0)) != nullptr) {
        delete child->widget();
        delete child;
    }
    if (mat.empty()) return;

    QLabel *label = new QLabel();
    QPixmap pix = scaleAndConvert(mat, 800, 600);
    label->setPixmap(pix);
    label->setAlignment(Qt::AlignCenter);
    displayLayout->addWidget(label);
}

void MainWindow::updateDisplay(const std::vector<cv::Mat>& images, const QStringList& titles) {
    // Clear layout
    QLayoutItem *child;
    while ((child = displayLayout->takeAt(0)) != nullptr) {
        delete child->widget();
        delete child;
    }
    if (images.empty()) return;

    int maxW = 400, maxH = 300;
    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].empty()) continue;
        QWidget *container = new QWidget();
        QVBoxLayout *vbox = new QVBoxLayout(container);
        QLabel *imgLabel = new QLabel();
        QPixmap pix = scaleAndConvert(images[i], maxW, maxH);
        imgLabel->setPixmap(pix);
        imgLabel->setAlignment(Qt::AlignCenter);
        vbox->addWidget(imgLabel);
        if (i < titles.size() && !titles[i].isEmpty()) {
            QLabel *titleLabel = new QLabel(titles[i]);
            titleLabel->setAlignment(Qt::AlignCenter);
            titleLabel->setStyleSheet("font-weight: bold; margin-top: 5px;");
            vbox->addWidget(titleLabel);
        }
        displayLayout->addWidget(container);
    }
}

// -------------------- File Operations --------------------
void MainWindow::handleLoad() {
    QString path = QFileDialog::getOpenFileName(this, "Select Image", "",
                                                "Images (*.png *.jpg *.jpeg *.bmp)");
    if (path.isEmpty()) return;
    originalFull = cv::imread(path.toStdString());
    if (originalFull.empty()) {
        QMessageBox::critical(this, "Error", "Could not load image!");
        return;
    }
    // Resize for processing
    originalResized = Utils::resizeAspect(originalFull, 512);
    processedMat = originalResized.clone();
    secondFull = cv::Mat();
    secondResized = cv::Mat();
    fftValid = false;
    secondFFTValid = false;
    updateDisplay(processedMat);
}

void MainWindow::handleSave() {
    if (processedMat.empty()) return;
    QString path = QFileDialog::getSaveFileName(this, "Save Image", "",
                                                "PNG (*.png);;JPG (*.jpg)");
    if (!path.isEmpty()) {
        cv::imwrite(path.toStdString(), processedMat);
    }
}

void MainWindow::handleReset() {
    if (originalResized.empty()) return;
    processedMat = originalResized.clone();
    updateDisplay(processedMat);
}

// -------------------- Noise --------------------
void MainWindow::handleUniformNoise() {
    if (originalResized.empty()) return;
    processedMat = Noise::addUniform(originalResized, 0, 50);
    updateDisplay(processedMat);
}

void MainWindow::handleGaussianNoise() {
    if (originalResized.empty()) return;
    processedMat = Noise::addGaussian(originalResized, 0, 25);
    updateDisplay(processedMat);
}

void MainWindow::handleSaltPepperNoise() {
    if (originalResized.empty()) return;
    processedMat = Noise::addSaltPepper(originalResized, 0.05);
    updateDisplay(processedMat);
}

void MainWindow::handleAllNoise() {
    if (originalResized.empty()) return;
    std::vector<cv::Mat> images;
    images.push_back(Noise::addUniform(originalResized, 0, 50));
    images.push_back(Noise::addGaussian(originalResized, 0, 25));
    images.push_back(Noise::addSaltPepper(originalResized, 0.05));
    QStringList titles = {"Uniform", "Gaussian", "Salt & Pepper"};
    updateDisplay(images, titles);
}

// -------------------- Filters --------------------
void MainWindow::handleAverageFilter() {
    if (originalResized.empty()) return;
    processedMat = Filters::average(originalResized, 3);
    updateDisplay(processedMat);
}

void MainWindow::handleGaussianFilter() {
    if (originalResized.empty()) return;
    processedMat = Filters::gaussian(originalResized, 3, 1.0);
    updateDisplay(processedMat);
}

void MainWindow::handleMedianFilter() {
    if (originalResized.empty()) return;
    processedMat = Filters::median(originalResized, 3);
    updateDisplay(processedMat);
}

void MainWindow::handleAllFilters() {
    if (originalResized.empty()) return;
    std::vector<cv::Mat> images;
    images.push_back(Filters::average(originalResized, 3));
    images.push_back(Filters::gaussian(originalResized, 3, 1.0));
    images.push_back(Filters::median(originalResized, 3));
    QStringList titles = {"Average", "Gaussian", "Median"};
    updateDisplay(images, titles);
}

// -------------------- Edge Detection --------------------
void MainWindow::handleSobel() {
    if (originalResized.empty()) return;
    processedMat = EdgeDetection::sobel(originalResized);
    updateDisplay(processedMat);
}

void MainWindow::handleRoberts() {
    if (originalResized.empty()) return;
    processedMat = EdgeDetection::roberts(originalResized);
    updateDisplay(processedMat);
}

void MainWindow::handlePrewitt() {
    if (originalResized.empty()) return;
    processedMat = EdgeDetection::prewitt(originalResized);
    updateDisplay(processedMat);
}

void MainWindow::handleCanny() {
    if (originalResized.empty()) return;
    processedMat = EdgeDetection::canny(originalResized, 100, 200);
    updateDisplay(processedMat);
}

void MainWindow::handleAllEdges() {
    if (originalResized.empty()) return;
    std::vector<cv::Mat> images;
    images.push_back(EdgeDetection::sobel(originalResized));
    images.push_back(EdgeDetection::roberts(originalResized));
    images.push_back(EdgeDetection::prewitt(originalResized));
    images.push_back(EdgeDetection::canny(originalResized, 100, 200));
    QStringList titles = {"Sobel", "Roberts", "Prewitt", "Canny"};
    updateDisplay(images, titles);
}

// -------------------- Histogram --------------------
void MainWindow::handleShowHistogramGray() {
    if (originalResized.empty()) return;
    processedMat = Histogram::computeHistogramImage(originalResized, false);
    updateDisplay(processedMat);
}

void MainWindow::handleShowHistogramRGB() {
    if (originalResized.empty()) return;
    processedMat = Histogram::computeRGBHistograms(originalResized, false);
    updateDisplay(processedMat);
}

void MainWindow::handleEqualize() {
    if (originalResized.empty()) return;
    processedMat = Histogram::equalize(originalResized);
    updateDisplay(processedMat);
}

void MainWindow::handleNormalize() {
    if (originalResized.empty()) return;
    processedMat = Histogram::normalize(originalResized);
    updateDisplay(processedMat);
}

// -------------------- Threshold --------------------
void MainWindow::handleGlobalThreshold() {
    if (originalResized.empty()) return;
    processedMat = Threshold::global(originalResized, 128);
    updateDisplay(processedMat);
}

void MainWindow::handleLocalThreshold() {
    if (originalResized.empty()) return;
    processedMat = Threshold::local(originalResized, 11, 2);
    updateDisplay(processedMat);
}

// -------------------- FFT Cache Helpers --------------------
void MainWindow::ensureFFTCached() {
    if (fftValid) return;
    if (originalResized.empty()) return;
    cv::Mat gray = Utils::toGrayscale(originalResized);
    cachedFFT = FrequencyDomain::computeFFT(gray);
    fftValid = true;
}

void MainWindow::ensureSecondFFTCached() {
    if (secondFFTValid) return;
    if (secondResized.empty()) return;
    cv::Mat gray = Utils::toGrayscale(secondResized);
    cachedFFTSecond = FrequencyDomain::computeFFT(gray);
    secondFFTValid = true;
}

// -------------------- Frequency Domain --------------------
void MainWindow::handleLowPass() {
    if (originalResized.empty()) return;
    ensureFFTCached();
    processedMat = FrequencyDomain::applyLowPass(cachedFFT, 30);
    updateDisplay(processedMat);
}

void MainWindow::handleHighPass() {
    if (originalResized.empty()) return;
    ensureFFTCached();
    processedMat = FrequencyDomain::applyHighPass(cachedFFT, 30);
    updateDisplay(processedMat);
}

// -------------------- Hybrid --------------------
void MainWindow::loadSecondImage() {
    QString path = QFileDialog::getOpenFileName(this, "Select Second Image", "",
                                                "Images (*.png *.jpg *.jpeg *.bmp)");
    if (path.isEmpty()) return;
    secondFull = cv::imread(path.toStdString());
    if (secondFull.empty()) {
        QMessageBox::critical(this, "Error", "Could not load second image!");
        return;
    }
    secondResized = Utils::resizeAspect(secondFull, 512);
    secondFFTValid = false; // invalidate cache

    // Display both original images side by side
    if (!originalResized.empty() && !secondResized.empty()) {
        std::vector<cv::Mat> both = {originalResized, secondResized};
        QStringList titles = {"Image 1", "Image 2"};
        updateDisplay(both, titles);
    }
}

void MainWindow::handleHybrid() {
    if (originalResized.empty() || secondResized.empty()) {
        QMessageBox::warning(this, "Warning", "Please load two images first!");
        return;
    }

    ensureFFTCached();
    ensureSecondFFTCached();

    cv::Mat low, high;
    if (hybridMode) {
        // Image1 low, Image2 high
        low = FrequencyDomain::applyLowPass(cachedFFT, 30);
        high = FrequencyDomain::applyHighPass(cachedFFTSecond, 30);
    } else {
        // Image1 high, Image2 low
        high = FrequencyDomain::applyHighPass(cachedFFT, 30);
        low = FrequencyDomain::applyLowPass(cachedFFTSecond, 30);
    }

    // Ensure same size (should be, but just in case)
    if (low.size() != high.size()) {
        cv::resize(high, high, low.size());
    }
    cv::Mat hybrid;
    cv::addWeighted(low, 0.5, high, 0.5, 0, hybrid);
    processedMat = hybrid;

    // Display all three: Image1, Image2, Hybrid
    std::vector<cv::Mat> three = {originalResized, secondResized, hybrid};
    QStringList titles = {"Image 1", "Image 2", "Hybrid"};
    updateDisplay(three, titles);
}