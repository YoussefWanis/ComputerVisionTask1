#include <QApplication>
#include "MainWindow.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    app.setWindowIcon(QIcon(":/icons/app_icon.png"));
    // Set application info
    QApplication::setApplicationName("Computer Vision Task 1");
    QApplication::setApplicationVersion("1.0");

    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}