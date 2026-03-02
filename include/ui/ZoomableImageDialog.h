#ifndef ZOOMABLEIMAGEDIALOG_H
#define ZOOMABLEIMAGEDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QScrollArea>
#include <QImage>
#include <QPixmap>
#include <QWheelEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QStatusBar>
#include <opencv2/opencv.hpp>

/**
 * ZoomableImageDialog -- opens an image in a separate resizable window.
 * Supports mouse-wheel zoom in/out and a Save button.
 */
class ZoomableImageDialog : public QDialog {
    Q_OBJECT
public:
    explicit ZoomableImageDialog(const QImage& image, const cv::Mat& mat,
                                 const QString& title = "Image Preview",
                                 QWidget* parent = nullptr)
        : QDialog(parent), fullImage_(image), cvMat_(mat.clone()), zoomFactor_(1.0)
    {
        setWindowTitle(title);
        resize(800, 600);
        setMinimumSize(400, 300);

        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(0, 0, 0, 0);

        // Toolbar
        auto* toolbar = new QHBoxLayout;
        toolbar->setContentsMargins(6, 4, 6, 4);

        auto* btnZoomIn = new QPushButton("+ Zoom In");
        auto* btnZoomOut = new QPushButton("- Zoom Out");
        auto* btnFit = new QPushButton("Fit to Window");
        auto* btnActual = new QPushButton("100%");
        auto* btnSave = new QPushButton("Save Image");
        zoomLabel_ = new QLabel("100%");
        zoomLabel_->setFixedWidth(60);

        connect(btnZoomIn,  &QPushButton::clicked, this, [this]() { zoom(1.25); });
        connect(btnZoomOut, &QPushButton::clicked, this, [this]() { zoom(0.8); });
        connect(btnFit,     &QPushButton::clicked, this, &ZoomableImageDialog::fitToWindow);
        connect(btnActual,  &QPushButton::clicked, this, [this]() {
            zoomFactor_ = 1.0;
            updateDisplay();
        });
        connect(btnSave, &QPushButton::clicked, this, &ZoomableImageDialog::saveImage);

        toolbar->addWidget(btnZoomIn);
        toolbar->addWidget(btnZoomOut);
        toolbar->addWidget(btnFit);
        toolbar->addWidget(btnActual);
        toolbar->addStretch();
        toolbar->addWidget(zoomLabel_);
        toolbar->addWidget(btnSave);
        root->addLayout(toolbar);

        // Scroll area with image
        scrollArea_ = new QScrollArea;
        scrollArea_->setWidgetResizable(false);
        scrollArea_->setAlignment(Qt::AlignCenter);
        scrollArea_->setStyleSheet("background: #2b2b2b;");

        imageLabel_ = new QLabel;
        imageLabel_->setAlignment(Qt::AlignCenter);
        scrollArea_->setWidget(imageLabel_);

        root->addWidget(scrollArea_, 1);

        fitToWindow();
    }

protected:
    void wheelEvent(QWheelEvent* event) override {
        if (event->angleDelta().y() > 0)
            zoom(1.15);
        else
            zoom(1.0 / 1.15);
        event->accept();
    }

    void resizeEvent(QResizeEvent* event) override {
        QDialog::resizeEvent(event);
        // Don't auto-fit on resize -- keep user's zoom level
    }

private slots:
    void saveImage() {
        QString path = QFileDialog::getSaveFileName(
            this, "Save Image", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All (*)");
        if (path.isEmpty()) return;
        if (!cvMat_.empty())
            cv::imwrite(path.toStdString(), cvMat_);
    }

private:
    void zoom(double factor) {
        zoomFactor_ *= factor;
        zoomFactor_ = std::max(0.05, std::min(zoomFactor_, 20.0));
        updateDisplay();
    }

    void fitToWindow() {
        if (fullImage_.isNull()) return;
        QSize viewSize = scrollArea_->viewport()->size();
        double scaleW = static_cast<double>(viewSize.width())  / fullImage_.width();
        double scaleH = static_cast<double>(viewSize.height()) / fullImage_.height();
        zoomFactor_ = std::min(scaleW, scaleH);
        if (zoomFactor_ <= 0) zoomFactor_ = 1.0;
        updateDisplay();
    }

    void updateDisplay() {
        if (fullImage_.isNull()) return;
        int w = static_cast<int>(fullImage_.width()  * zoomFactor_);
        int h = static_cast<int>(fullImage_.height() * zoomFactor_);
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        QPixmap pix = QPixmap::fromImage(fullImage_)
                          .scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        imageLabel_->setPixmap(pix);
        imageLabel_->resize(pix.size());
        zoomLabel_->setText(QString("%1%").arg(static_cast<int>(zoomFactor_ * 100)));
    }

    QImage      fullImage_;
    cv::Mat     cvMat_;
    double      zoomFactor_;
    QScrollArea* scrollArea_  = nullptr;
    QLabel*      imageLabel_  = nullptr;
    QLabel*      zoomLabel_   = nullptr;
};

#endif // ZOOMABLEIMAGEDIALOG_H
