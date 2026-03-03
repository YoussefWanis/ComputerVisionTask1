cmake_minimum_required(VERSION 3.14)

# Define the project name and language (C++)
project(ComputerVisionTask1 LANGUAGES CXX)

# Enforce C++17 as the language standard across all targets
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTORCC ON)

# Enable Qt's Meta-Object Compiler (MOC) so that Q_OBJECT macros
# in headers are automatically processed during the build
set(CMAKE_AUTOMOC ON)

# ── Dependencies ─────────────────────────────────────────────────
# Locate OpenCV (provides ${OpenCV_LIBS} and ${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)

# Locate Qt6 Widgets module (provides Qt6::Widgets link target)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# ── Include paths ────────────────────────────────────────────────
# Make project headers and OpenCV headers available to all targets
include_directories(
    ${CMAKE_SOURCE_DIR}/include   # Project headers (models, processors, utils, ui, metrics)
    ${OpenCV_INCLUDE_DIRS}        # OpenCV headers
)

# ── Source files ─────────────────────────────────────────────────
# All .cpp implementation files grouped by logical module
set(SOURCES
        resources.qrc
    # Models — application data layer (loaded image + result cache)
    src/models/ImageModel.cpp

    # Utils — shared low-level helpers (padding, correlation, FFT shift)
    src/utils/ImageUtils.cpp

    # Processors — core image-processing algorithms
    src/processors/ColorProcessor.cpp          # BGR↔Grayscale, channel splitting
    src/processors/NoiseProcessor.cpp          # Uniform, Gaussian, Salt & Pepper noise
    src/processors/FilterProcessor.cpp         # Average, Gaussian, Median spatial filters
    src/processors/EdgeDetectorProcessor.cpp   # Sobel, Roberts, Prewitt, Canny edges
    src/processors/HistogramProcessor.cpp      # Histogram equalisation, normalisation
    src/processors/FFTProcessor.cpp            # Frequency-domain low/high-pass filtering
    src/processors/HybridProcessor.cpp         # Hybrid image creation via FFT

    # Metrics — image quality assessment (MSE, PSNR, SNR)
    src/metrics/MetricsCalculator.cpp

    # UI — Qt GUI (one file per tab + main window core)
    src/ui/MainWindow.cpp                # Window setup, image loading, display helpers
    src/ui/MainWindow_NoiseFilter.cpp    # Tab 1: Noise & Filter controls/slots
    src/ui/MainWindow_Edge.cpp           # Tab 2: Edge Detection controls/slots
    src/ui/MainWindow_Histogram.cpp      # Tab 3: Histograms & Colour controls/slots
    src/ui/MainWindow_FFT.cpp            # Tab 4: Frequency Domain controls/slots
    src/ui/MainWindow_Hybrid.cpp         # Tab 5: Hybrid Images controls/slots

    # Entry point — creates the QApplication and shows MainWindow
    src/main.cpp
)

# All header files (listed explicitly so AUTOMOC processes Q_OBJECT headers)
set(HEADERS
    include/models/ImageModel.h
    include/utils/ImageUtils.h
    include/processors/ColorProcessor.h
    include/processors/NoiseProcessor.h
    include/processors/FilterProcessor.h
    include/processors/EdgeDetectorProcessor.h
    include/processors/HistogramProcessor.h
    include/processors/FFTProcessor.h
    include/processors/HybridProcessor.h
    include/metrics/MetricsCalculator.h
    include/ui/MainWindow.h
    include/ui/ZoomableImageDialog.h
)

# ── Build target ─────────────────────────────────────────────────
# Create the executable from all source and header files
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link against OpenCV libraries and Qt6 Widgets
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}     # OpenCV core, imgproc, imgcodecs, etc.
    Qt6::Widgets       # Qt widgets, core, gui
)
