#include <QApplication>
#include "ui/MainWindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setStyle("Fusion");
    QApplication::setApplicationName("CV Task 1 — Image Processing Pipeline");

    MainWindow win;
    win.show();

    return app.exec();
}
