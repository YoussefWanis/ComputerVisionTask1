/**
 * @file ZoomableImageDialog.h
 * @brief Declaration and inline implementation of ZoomableImageDialog,
 *        a pop-up window for previewing images with zoom and save.
 *
 * Opened when the user double-clicks an image label in the main window.
 * Features:
 *   - Mouse-wheel zoom in/out (1.15× per scroll notch).
 *   - Toolbar buttons: Zoom In, Zoom Out, Fit to Window, 100%, Save.
 *   - Scroll bars for panning when zoomed in beyond the window size.
 *   - Saves via cv::imwrite, preserving the original cv::Mat quality.
 */

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
 * @class ZoomableImageDialog
 * @brief A resizable dialog that displays a full-resolution image
 *        with interactive zoom controls and a save button.
 *
 * The dialog receives both a QImage (for display) and the original
 * cv::Mat (for high-quality saving).  Zoom level is tracked as a
 * floating-point factor clamped to [0.05, 20.0].
 */
class ZoomableImageDialog : public QDialog {
    Q_OBJECT
public:
    /**
     * @brief Construct the dialog and lay out its UI.
     *
     * @param image   Full-resolution QImage to display.
     * @param mat     Original OpenCV Mat (cloned internally) used
     *                for saving to disk at full quality.
     * @param title   Window title shown in the title bar.
     * @param parent  Parent widget (typically the MainWindow).
     */
    explicit ZoomableImageDialog(const QImage& image, const cv::Mat& mat,
                                 const QString& title = "Image Preview",
                                 QWidget* parent = nullptr)
        : QDialog(parent), fullImage_(image), cvMat_(mat.clone()), zoomFactor_(1.0)
    {
        setWindowTitle(title);
        resize(800, 600);
        setMinimumSize(400, 300);

        // Root vertical layout for the entire dialog
        auto* root = new QVBoxLayout(this);
        root->setContentsMargins(0, 0, 0, 0);

        // ?? Toolbar row ?????????????????????????????????????????
        auto* toolbar = new QHBoxLayout;
        toolbar->setContentsMargins(6, 4, 6, 4);

        auto* btnZoomIn  = new QPushButton("+ Zoom In");
        auto* btnZoomOut = new QPushButton("- Zoom Out");
        auto* btnFit     = new QPushButton("Fit to Window");
        auto* btnActual  = new QPushButton("100%");
        auto* btnSave    = new QPushButton("Save Image");

        // Label displaying the current zoom percentage
        zoomLabel_ = new QLabel("100%");
        zoomLabel_->setFixedWidth(60);

        // Connect toolbar buttons to zoom / save actions
        connect(btnZoomIn,  &QPushButton::clicked, this, [this]() { zoom(1.25); });
        connect(btnZoomOut, &QPushButton::clicked, this, [this]() { zoom(0.8); });
        connect(btnFit,     &QPushButton::clicked, this, &ZoomableImageDialog::fitToWindow);
        connect(btnActual,  &QPushButton::clicked, this, [this]() {
            zoomFactor_ = 1.0;   // Reset to native resolution
            updateDisplay();
        });
        connect(btnSave, &QPushButton::clicked, this, &ZoomableImageDialog::saveImage);

        toolbar->addWidget(btnZoomIn);
        toolbar->addWidget(btnZoomOut);
        toolbar->addWidget(btnFit);
        toolbar->addWidget(btnActual);
        toolbar->addStretch();         // Push zoom label & save to the right
        toolbar->addWidget(zoomLabel_);
        toolbar->addWidget(btnSave);
        root->addLayout(toolbar);

        // ?? Scroll area containing the image label ??????????????
        scrollArea_ = new QScrollArea;
        scrollArea_->setWidgetResizable(false);          // We manage sizing ourselves
        scrollArea_->setAlignment(Qt::AlignCenter);      // Centre the image in the viewport
        scrollArea_->setStyleSheet("background: #2b2b2b;");  // Dark background

        imageLabel_ = new QLabel;
        imageLabel_->setAlignment(Qt::AlignCenter);
        scrollArea_->setWidget(imageLabel_);

        root->addWidget(scrollArea_, 1);   // Scroll area fills remaining space

        // Start by fitting the image to the current window size
        fitToWindow();
    }

protected:
    /**
     * @brief Handle mouse-wheel events to zoom in/out.
     *
     * Scrolling up zooms in by 15%, scrolling down zooms out by 15%.
     *
     * @param event  The wheel event containing scroll direction.
     */
    void wheelEvent(QWheelEvent* event) override {
        if (event->angleDelta().y() > 0)
            zoom(1.15);           // Scroll up ? zoom in
        else
            zoom(1.0 / 1.15);    // Scroll down ? zoom out
        event->accept();
    }

    /**
     * @brief Handle dialog resize.  Keeps the user's zoom level
     *        (does not auto-fit on resize).
     */
    void resizeEvent(QResizeEvent* event) override {
        QDialog::resizeEvent(event);
        // Intentionally do not call fitToWindow() here —
        // preserve the zoom level the user has chosen.
    }

private slots:
    /**
     * @brief Open a Save dialog and write the original cv::Mat to disk.
     *
     * Uses cv::imwrite so the saved image has the same quality as the
     * original data (not the scaled/displayed version).
     */
    void saveImage() {
        QString path = QFileDialog::getSaveFileName(
            this, "Save Image", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All (*)");
        if (path.isEmpty()) return;
        if (!cvMat_.empty())
            cv::imwrite(path.toStdString(), cvMat_);
    }

private:
    /**
     * @brief Multiply the current zoom factor and refresh the display.
     *
     * The zoom level is clamped to [0.05, 20.0] to prevent extreme
     * scaling.
     *
     * @param factor  Multiplicative zoom factor (>1 = zoom in,
     *                <1 = zoom out).
     */
    void zoom(double factor) {
        zoomFactor_ *= factor;
        zoomFactor_ = std::max(0.05, std::min(zoomFactor_, 20.0));
        updateDisplay();
    }

    /**
     * @brief Calculate and apply a zoom factor that fits the entire
     *        image within the scroll area's visible viewport.
     */
    void fitToWindow() {
        if (fullImage_.isNull()) return;
        QSize viewSize = scrollArea_->viewport()->size();
        double scaleW = static_cast<double>(viewSize.width())  / fullImage_.width();
        double scaleH = static_cast<double>(viewSize.height()) / fullImage_.height();
        zoomFactor_ = std::min(scaleW, scaleH);   // Fit the larger dimension
        if (zoomFactor_ <= 0) zoomFactor_ = 1.0;
        updateDisplay();
    }

    /**
     * @brief Re-render the image at the current zoom level.
     *
     * Scales the full-resolution QImage to (width × zoom, height × zoom),
     * sets the resulting pixmap on the label, and updates the zoom
     * percentage indicator.
     */
    void updateDisplay() {
        if (fullImage_.isNull()) return;

        // Compute the target pixel dimensions
        int w = static_cast<int>(fullImage_.width()  * zoomFactor_);
        int h = static_cast<int>(fullImage_.height() * zoomFactor_);
        if (w < 1) w = 1;
        if (h < 1) h = 1;

        // Scale with smooth (bilinear) interpolation
        QPixmap pix = QPixmap::fromImage(fullImage_)
                          .scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        imageLabel_->setPixmap(pix);
        imageLabel_->resize(pix.size());

        // Update the zoom percentage label (e.g. "150%")
        zoomLabel_->setText(QString("%1%").arg(static_cast<int>(zoomFactor_ * 100)));
    }

    QImage       fullImage_;                ///< Full-resolution display image (QImage).
    cv::Mat      cvMat_;                    ///< Original OpenCV Mat (for saving).
    double       zoomFactor_;               ///< Current zoom multiplier (1.0 = 100%).
    QScrollArea* scrollArea_  = nullptr;    ///< Scroll area providing pan capability.
    QLabel*      imageLabel_  = nullptr;    ///< Label that displays the scaled pixmap.
    QLabel*      zoomLabel_   = nullptr;    ///< Toolbar label showing zoom percentage.
};

#endif // ZOOMABLEIMAGEDIALOG_H
