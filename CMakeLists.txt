cmake_minimum_required(VERSION 3.20)
project(ComputerVisionTask1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
        src/main.cpp
        src/MainWindow.cpp
        src/Utils.cpp
        src/Noise.cpp
        src/Filters.cpp
        src/EdgeDetection.cpp
        src/Histogram.cpp
        src/Threshold.cpp
        src/FrequencyDomain.cpp
        src/Hybrid.cpp
)

set(HEADERS
        include/ImageProcessor.h
        include/MainWindow.h
        include/Utils.h
        include/Noise.h
        include/Filters.h
        include/EdgeDetection.h
        include/Histogram.h
        include/Threshold.h
        include/FrequencyDomain.h
        include/Hybrid.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} resources.qrc  app.rc)
target_link_libraries(${PROJECT_NAME} Qt6::Core Qt6::Widgets ${OpenCV_LIBS})

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES WIN32_EXECUTABLE TRUE)
endif()