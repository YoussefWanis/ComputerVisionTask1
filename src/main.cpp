/**
 * @file main.cpp
 * @brief Application entry point for the Computer Vision Task 1 GUI.
 *
 * Creates a Qt application instance, configures its style and name,
 * then launches the main window which contains all image-processing
 * tabs (Noise & Filter, Edge Detection, Histograms, FFT, Hybrid).
 */

#include <QApplication>
#include "ui/MainWindow.h"

/**
 * @brief Program entry point.
 *
 * @param argc  Number of command-line arguments.
 * @param argv  Array of command-line argument strings.
 * @return      Exit code returned by the Qt event loop.
 */
int main(int argc, char* argv[]) {
    // Initialize the Qt application framework with command-line args
    QApplication app(argc, argv);

    // Use the "Fusion" style for a consistent cross-platform look
    app.setStyle("Fusion");

    // Set the application name displayed in window titles / task bars
    QApplication::setApplicationName("CV Task 1 — Image Processing Pipeline");

    // Create and display the main window (all tabs are built in its constructor)
    MainWindow win;
    win.show();

    // Enter the Qt event loop; blocks until the window is closed
    return app.exec();
}
