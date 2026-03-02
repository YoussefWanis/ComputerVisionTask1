cmake_minimum_required(VERSION 3.14)
project(CVTask1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)          # Qt Meta-Object Compiler

# ── Dependencies ─────────────────────────────────────────────────
set(OpenCV_DIR "D:/opencv/build")
set(CMAKE_PREFIX_PATH "D:/QT/6.10.2/msvc2022_64/lib/cmake")
find_package(OpenCV REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# ── Include paths ────────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ── Source files ─────────────────────────────────────────────────
set(SOURCES
    # Models
    src/models/ImageModel.cpp

    # Utils (shared helpers)
    src/utils/ImageUtils.cpp

    # Processors
    src/processors/ColorProcessor.cpp
    src/processors/NoiseProcessor.cpp
    src/processors/FilterProcessor.cpp
    src/processors/EdgeDetectorProcessor.cpp
    src/processors/HistogramProcessor.cpp
    src/processors/FFTProcessor.cpp
    src/processors/HybridProcessor.cpp

    # Metrics
    src/metrics/MetricsCalculator.cpp

    # UI
    src/ui/MainWindow.cpp
    src/ui/MainWindow_NoiseFilter.cpp
    src/ui/MainWindow_Edge.cpp
    src/ui/MainWindow_Histogram.cpp
    src/ui/MainWindow_FFT.cpp
    src/ui/MainWindow_Hybrid.cpp

    # Entry point
    src/main.cpp
)

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
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    Qt6::Widgets
)
