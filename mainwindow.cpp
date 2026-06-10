#include "MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QPainter>
#include <QPen>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>

namespace fs = std::filesystem;

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    engine = new FacePCA(50);
    isModelTrained = false;

    face_cascade.load("D:/Lyan/3rd year/2nd term/CV/task5/haarcascade_frontalface_default.xml");

    this->setStyleSheet("background-color: #121212; color: white; font-family: 'Segoe UI';");
    this->setWindowTitle("Final Project: Face Detection & Recognition");
    this->resize(1200, 700);

    setupLayout();
}

MainWindow::~MainWindow() {
    delete engine;
}

void MainWindow::setupLayout() {
    QWidget *central = new QWidget();
    QVBoxLayout *mainLayout = new QVBoxLayout(central);

    QLabel *header = new QLabel("CV PROJECT: DETECTION & RECOGNITION");
    header->setStyleSheet("font-size: 26px; font-weight: bold; color: #deff9a; padding: 10px;");
    header->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(header);

    tabs = new QTabWidget();
    tabs->setStyleSheet("QTabBar::tab { background: #333; color: white; padding: 15px; font-weight: bold; font-size: 14px;} "
                        "QTabBar::tab:selected { background: #deff9a; color: black; } "
                        "QTabWidget::pane { border: 2px solid #333; }");

    QWidget *tab1 = new QWidget();
    QWidget *tab2 = new QWidget();
    QWidget *tab3 = new QWidget();

    setupDetectionTab(tab1);
    setupRecognitionTab(tab2);
    setupPerformanceTab(tab3);

    tabs->addTab(tab1, "1. Face Detection");
    tabs->addTab(tab2, "2. Face Recognition (PCA)");
    tabs->addTab(tab3, "3. Performance & ROC");

    mainLayout->addWidget(tabs);
    setCentralWidget(central);
}
// ==========================================
// TAB 1: FACE DETECTION
// ==========================================
void MainWindow::setupDetectionTab(QWidget *tab) {
    QVBoxLayout *layout = new QVBoxLayout(tab);

    // Label to display the image and detection results
    imgDetection = new QLabel("Upload an image to detect faces");
    imgDetection->setAlignment(Qt::AlignCenter);
    imgDetection->setStyleSheet("border: 2px dashed #555; background: #1a1a1a;");
    imgDetection->setMinimumHeight(400);
    layout->addWidget(imgDetection);

    // Buttons for loading image and running detection
    QHBoxLayout *btnLayout = new QHBoxLayout();

    btnLoadDetImage = new QPushButton("Load Image");
    btnRunDetection = new QPushButton("Detect Faces");

    QString btnStyle =
        "QPushButton { background-color: #444; border: none; padding: 12px; font-weight: bold; border-radius: 5px; } "
        "QPushButton:hover { background-color: #555; }";

    btnLoadDetImage->setStyleSheet(btnStyle);
    btnRunDetection->setStyleSheet(btnStyle);

    btnLayout->addWidget(btnLoadDetImage);
    btnLayout->addWidget(btnRunDetection);
    layout->addLayout(btnLayout);

    // Connect buttons to corresponding handlers
    connect(btnLoadDetImage, SIGNAL(clicked()), this, SLOT(handleLoadDetectionImage()));
    connect(btnRunDetection, SIGNAL(clicked()), this, SLOT(handleRunDetection()));
}

/**
 * Handles loading an image from disk for face detection.
 * - Opens file dialog
 * - Loads image using OpenCV
 * - Displays it in the UI
 */
void MainWindow::handleLoadDetectionImage() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select Image",
        "D:/Lyan/3rd year/2nd term/CV/task5/dataset",
        "Images (*.png *.jpg *.jpeg *.pgm)"
        );

    if (fileName.isEmpty()) return;

    // Load image in BGR format
    currentDetectionImage = cv::imread(fileName.toStdString());

    // Display original image
    displayImageOnLabel(currentDetectionImage, imgDetection);
}

/**
 * Handles face detection:
 * - Converts image to grayscale
 * - Applies histogram equalization
 * - Detects faces using Haar Cascade
 * - Draws bounding boxes on detected faces
 */
void MainWindow::handleRunDetection() {
    // Ensure an image is loaded
    if (currentDetectionImage.empty()) return;

    cv::Mat gray;
    cv::Mat resultImage;

    // Copy original image for drawing results
    currentDetectionImage.copyTo(resultImage);

    // Convert to grayscale if needed
    if (resultImage.channels() == 3)
        cv::cvtColor(resultImage, gray, cv::COLOR_BGR2GRAY);
    else
        resultImage.copyTo(gray);

    // Improve contrast using histogram equalization
    cv::equalizeHist(gray, gray);

    // Detect faces
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(
        gray,
        faces,
        1.1,                  // Scale factor
        7,                    // Min neighbors
        0 | cv::CASCADE_SCALE_IMAGE,
        cv::Size(50, 50)      // Minimum face size
        );

    // Draw bounding boxes around detected faces
    for (const auto& face : faces) {
        cv::rectangle(resultImage, face, cv::Scalar(0, 255, 0), 3);
    }

    // Display detection result
    displayImageOnLabel(resultImage, imgDetection);
}

// ==========================================
// TAB 2: FACE RECOGNITION (PCA)
// ==========================================
void MainWindow::setupRecognitionTab(QWidget *tab) {
    QVBoxLayout *layout = new QVBoxLayout(tab);

    QHBoxLayout *contentLayout = new QHBoxLayout();

    // Label to display the input (probe) image
    imgInput = new QLabel("Probe Image");
    imgInput->setFixedSize(300, 350);
    imgInput->setAlignment(Qt::AlignCenter);
    imgInput->setStyleSheet("border: 2px dashed #555; background: #1a1a1a;");

    // Label to display the matched result image
    imgResult = new QLabel("Matched Subject");
    imgResult->setFixedSize(300, 350);
    imgResult->setAlignment(Qt::AlignCenter);
    imgResult->setStyleSheet("border: 2px solid #deff9a; background: #1a1a1a;");

    contentLayout->addWidget(imgInput);
    contentLayout->addWidget(imgResult);
    layout->addLayout(contentLayout);

    // Buttons for training and recognition
    QHBoxLayout *btnLayout = new QHBoxLayout();

    btnTrain = new QPushButton("1. Train PCA Dataset");
    btnRecognize = new QPushButton("2. Test Face Match");

    QString btnStyle =
        "QPushButton { background-color: #444; border: none; padding: 12px; font-weight: bold; border-radius: 5px; } "
        "QPushButton:hover { background-color: #555; border: 1px solid #deff9a; }";

    btnTrain->setStyleSheet(btnStyle);
    btnRecognize->setStyleSheet(btnStyle);

    btnLayout->addWidget(btnTrain);
    btnLayout->addWidget(btnRecognize);
    layout->addLayout(btnLayout);

    // Status label for feedback
    lblStatus = new QLabel("Recognition Status: Waiting for Training...");
    lblStatus->setAlignment(Qt::AlignCenter);
    lblStatus->setStyleSheet("color: #aaa; font-size: 15px; font-weight: bold; padding: 10px;");
    layout->addWidget(lblStatus);

    // Connect buttons to corresponding slots
    connect(btnTrain, SIGNAL(clicked()), this, SLOT(handleTrainPCA()));
    connect(btnRecognize, SIGNAL(clicked()), this, SLOT(handleRecognizeFace()));
}

/**
 * Handles PCA training:
 * - Loads dataset from disk
 * - Preprocesses images
 * - Shuffles data
 * - Trains PCA model
 */
void MainWindow::handleTrainPCA() {
    lblStatus->setText("Status: Loading dataset and training PCA... Please wait.");
    QApplication::processEvents();

    std::string dataset_path = "D:/Lyan/3rd year/2nd term/CV/task5/dataset";

    cv::Mat X_train;
    std::vector<int> y_train;

    // Load dataset (subjects s1 → s10, images 1 → 7)
    for (int subject = 1; subject <= 10; ++subject) {
        std::string subject_path = dataset_path + "/s" + std::to_string(subject);
        if (!fs::exists(subject_path)) continue;

        for (int i = 1; i <= 7; ++i) {
            std::string img_path = subject_path + "/" + std::to_string(i) + ".pgm";
            if (!fs::exists(img_path)) continue;

            // Preprocess image and extract feature vector
            cv::Mat face_vector = engine->detectAndPreprocess(img_path, face_cascade);

            if (!face_vector.empty()) {
                X_train.push_back(face_vector);
                y_train.push_back(subject);
            }
        }
    }

    // Handle empty dataset case
    if (X_train.empty()) {
        lblStatus->setText("Error: Dataset not found or no faces detected!");
        return;
    }

    // Shuffle training data for better generalization
    int num_samples = X_train.rows;
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    cv::Mat X_train_shuffled;
    std::vector<int> y_train_shuffled;

    for (int i = 0; i < num_samples; ++i) {
        X_train_shuffled.push_back(X_train.row(indices[i]));
        y_train_shuffled.push_back(y_train[indices[i]]);
    }

    // Train PCA model
    engine->train(X_train_shuffled, y_train_shuffled);
    isModelTrained = true;

    // Update status
    lblStatus->setStyleSheet("color: #deff9a; font-size: 15px; font-weight: bold; padding: 10px;");
    lblStatus->setText("Status: Training complete for 10 subjects. Ready for recognition.");
}

/**
 * Handles face recognition:
 * - Loads probe image
 * - Preprocesses and projects it
 * - Predicts closest match
 * - Displays result and distance
 */
void MainWindow::handleRecognizeFace() {
    // Ensure model is trained
    if (!isModelTrained) {
        QMessageBox::warning(this, "Warning", "Please train the PCA model first (Step 1).");
        return;
    }

    // Select probe image
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select Probe Image",
        "D:/Lyan/3rd year/2nd term/CV/task5_media/test_dataset",
        "Images (*.pgm *.jpg *.png)"
        );

    if (fileName.isEmpty()) return;

    std::string img_path = fileName.toStdString();

    // Display input image
    displayImageOnLabel(
        cv::imread(img_path, cv::IMREAD_GRAYSCALE),
        imgInput
        );

    lblStatus->setText("Analyzing face...");
    QApplication::processEvents();

    // Preprocess image
    cv::Mat face_vector = engine->detectAndPreprocess(img_path, face_cascade);

    if (face_vector.empty()) {
        lblStatus->setText("Status: No face detected!");
        lblStatus->setStyleSheet("color: red; font-size: 15px; font-weight: bold; padding: 10px;");
        return;
    }

    // Project into PCA space
    cv::Mat test_weight = engine->project(face_vector);

    // Predict label
    double raw_distance;
    int predicted_label = engine->predict(test_weight.row(0), raw_distance);

    // Load and display matched subject image
    std::string matchPath =
        "D:/Lyan/3rd year/2nd term/CV/task5/dataset/s" +
        std::to_string(predicted_label) + "/1.pgm";

    displayImageOnLabel(
        cv::imread(matchPath, cv::IMREAD_GRAYSCALE),
        imgResult
        );

    // Display result with distance
    lblStatus->setText(
        QString("Status: Match found! Subject S%1 (Distance: %2)")
            .arg(predicted_label)
            .arg(raw_distance, 0, 'f', 2)
        );

    lblStatus->setStyleSheet("color: #deff9a; font-size: 15px; font-weight: bold; padding: 10px;");
}

// ==========================================
// TAB 3: PERFORMANCE & SIDE-BY-SIDE ROC CURVES
// ==========================================
void MainWindow::setupPerformanceTab(QWidget *tab) {
    QVBoxLayout *layout = new QVBoxLayout(tab);

    // Accuracy label (initial state)
    lblAccuracy = new QLabel("System Accuracy: N/A");
    lblAccuracy->setAlignment(Qt::AlignCenter);
    lblAccuracy->setStyleSheet("font-size: 28px; font-weight: bold; color: #deff9a;");
    layout->addWidget(lblAccuracy);

    // Layout to hold both ROC graphs side by side
    QHBoxLayout *graphsLayout = new QHBoxLayout();

    // ===== Global ROC Curve Section =====
    QVBoxLayout *g1Layout = new QVBoxLayout();

    QLabel *l1 = new QLabel("Overall System ROC (Genuine vs Impostor)");
    l1->setAlignment(Qt::AlignCenter);
    l1->setStyleSheet("font-weight: bold; color: #fff; font-size: 14px;");

    // Placeholder label for global ROC visualization
    imgGlobalRoc = new QLabel("Global Curve Area");
    imgGlobalRoc->setAlignment(Qt::AlignCenter);
    imgGlobalRoc->setStyleSheet("border: 1px solid #555; background: #1a1a1a;");
    imgGlobalRoc->setFixedSize(500, 400);

    g1Layout->addWidget(l1);
    g1Layout->addWidget(imgGlobalRoc);

    // ===== Multi-class ROC Curve Section =====
    QVBoxLayout *g2Layout = new QVBoxLayout();

    QLabel *l2 = new QLabel("Multi-class ROC (One-vs-Rest)");
    l2->setAlignment(Qt::AlignCenter);
    l2->setStyleSheet("font-weight: bold; color: #fff; font-size: 14px;");

    // Placeholder label for multi-class ROC visualization
    imgMultiRoc = new QLabel("Multi-class Area");
    imgMultiRoc->setAlignment(Qt::AlignCenter);
    imgMultiRoc->setStyleSheet("border: 1px solid #555; background: #1a1a1a;");
    imgMultiRoc->setFixedSize(500, 400);

    g2Layout->addWidget(l2);
    g2Layout->addWidget(imgMultiRoc);

    // Add both graph sections to the main horizontal layout
    graphsLayout->addLayout(g1Layout);
    graphsLayout->addLayout(g2Layout);
    layout->addLayout(graphsLayout);

    // Evaluation button
    btnEvaluate = new QPushButton("Select Test Dataset Folder & Evaluate");
    QString btnStyle =
        "QPushButton { background-color: #444; border: none; padding: 15px; font-weight: bold; border-radius: 5px; } "
        "QPushButton:hover { background-color: #555; border: 1px solid #00ffff; }";

    btnEvaluate->setStyleSheet(btnStyle);
    layout->addWidget(btnEvaluate);

    // Connect button click to evaluation handler
    connect(btnEvaluate, SIGNAL(clicked()), this, SLOT(handleEvaluate()));
}

/**
 * Handles model evaluation:
 * - Loads test dataset
 * - Runs predictions
 * - Computes accuracy
 * - Generates ROC curves (global + multi-class)
 */
void MainWindow::handleEvaluate() {
    // Ensure model is trained before evaluation
    if (!isModelTrained) {
        QMessageBox::warning(this, "Error", "Please train the PCA model first (in Tab 2) before evaluating!");
        return;
    }

    // Open folder selection dialog
    QString dir = QFileDialog::getExistingDirectory(
        this,
        "Select Testing Dataset Folder",
        "D:/Lyan/3rd year/2nd term/CV/task5_media"
        );

    if (dir.isEmpty()) return;

    // Update UI to indicate processing
    lblAccuracy->setText("Evaluating... Please Wait");
    QApplication::processEvents();

    // Structure to store test results
    struct TestSample {
        int true_label;
        int pred_label;
        double distance;
    };

    std::vector<TestSample> all_tests;

    int correct = 0, total = 0;
    double max_dist = 0;

    // Iterate over subjects (assumes folders s1 → s10)
    for (int subject = 1; subject <= 10; ++subject) {
        std::string path = dir.toStdString() + "/s" + std::to_string(subject);
        if (!fs::exists(path)) continue;

        for (const auto& entry : fs::directory_iterator(path)) {
            std::string img_path = entry.path().string();

            // Skip unsupported file formats
            if (img_path.find(".pgm") == std::string::npos &&
                img_path.find(".jpg") == std::string::npos &&
                img_path.find(".png") == std::string::npos)
                continue;

            // Preprocess image and extract feature vector
            cv::Mat face_vector = engine->detectAndPreprocess(img_path, face_cascade);

            if (!face_vector.empty()) {
                double raw_dist;

                // Predict label and compute distance
                int pred = engine->predict(
                    engine->project(face_vector).row(0),
                    raw_dist
                    );

                // Store result
                all_tests.push_back({subject, pred, raw_dist});

                if (pred == subject) correct++;
                total++;

                if (raw_dist > max_dist) max_dist = raw_dist;
            }
        }
    }

    // Handle case of no valid data
    if (total == 0) {
        lblAccuracy->setText("Error: No valid test images found!");
        return;
    }

    // ===== Global ROC Curve =====
    int steps = 100;
    std::vector<std::pair<double, double>> global_points;

    for (int i = 0; i <= steps; ++i) {
        double threshold = (max_dist / steps) * i;

        int tp = 0, fp = 0;
        int genuine_count = 0, impostor_count = 0;

        for (const auto& ts : all_tests) {
            if (ts.true_label == ts.pred_label) genuine_count++;
            else impostor_count++;

            if (ts.distance <= threshold) {
                if (ts.true_label == ts.pred_label) tp++;
                else fp++;
            }
        }

        double tpr = (genuine_count > 0) ? (double)tp / genuine_count : 0;
        double fpr = (impostor_count > 0) ? (double)fp / impostor_count : 0;

        global_points.push_back({fpr, tpr});
    }

    std::sort(global_points.begin(), global_points.end());
    drawGlobalROC(global_points);

    // ===== Multi-class ROC (One-vs-Rest) =====
    std::vector<std::vector<std::pair<double, double>>> multi_curves;

    for (int class_id = 1; class_id <= 10; ++class_id) {
        std::vector<std::pair<double, double>> curve;

        for (int i = 0; i <= steps; ++i) {
            double threshold = (max_dist / steps) * i;

            int tp = 0, fp = 0;
            int pos_total = 0, neg_total = 0;

            for (const auto& ts : all_tests) {
                if (ts.true_label == class_id) {
                    pos_total++;
                    if (ts.pred_label == class_id && ts.distance <= threshold) tp++;
                } else {
                    neg_total++;
                    if (ts.pred_label == class_id && ts.distance <= threshold) fp++;
                }
            }

            double tpr = (pos_total > 0) ? (double)tp / pos_total : 0;
            double fpr = (neg_total > 0) ? (double)fp / neg_total : 0;

            curve.push_back({fpr, tpr});
        }

        std::sort(curve.begin(), curve.end());
        multi_curves.push_back(curve);
    }

    drawMultiClassROC(multi_curves);

    // ===== Accuracy Calculation =====
    double acc = ((double)correct / total) * 100.0;
    lblAccuracy->setText(QString("System Accuracy: %1%").arg(QString::number(acc, 'f', 2)));
}

/**
 * Draws the global ROC curve (Genuine vs Impostor)
 */
void MainWindow::drawGlobalROC(const std::vector<std::pair<double, double>>& points) {
    QPixmap pix(imgGlobalRoc->size());
    pix.fill(QColor("#1a1a1a"));

    QPainter p(&pix);
    p.setRenderHint(QPainter::Antialiasing);

    int pad = 40;
    int w = pix.width() - 2 * pad;
    int h = pix.height() - 2 * pad;

    // Draw axes
    p.setPen(QPen(Qt::white, 2));
    p.drawLine(pad, pix.height() - pad, pix.width() - pad, pix.height() - pad);
    p.drawLine(pad, pad, pad, pix.height() - pad);

    // Axis labels
    p.drawText(pix.width() / 2 - 10, pix.height() - 10, "FPR");

    p.translate(15, pix.height() / 2 + 10);
    p.rotate(-90);
    p.drawText(0, 0, "TPR");
    p.rotate(90);
    p.translate(-15, -(pix.height() / 2 + 10));

    // Diagonal reference line
    p.setPen(QPen(Qt::gray, 1, Qt::DashLine));
    p.drawLine(pad, pix.height() - pad, pix.width() - pad, pad);

    // Plot ROC curve
    p.setPen(QPen(QColor("#00ffff"), 3));
    for (size_t i = 1; i < points.size(); ++i) {
        p.drawLine(
            pad + points[i - 1].first * w,
            (pix.height() - pad) - points[i - 1].second * h,
            pad + points[i].first * w,
            (pix.height() - pad) - points[i].second * h
            );
    }

    imgGlobalRoc->setPixmap(pix);
}

/**
 * Draws multi-class ROC curves (One-vs-Rest)
 */
void MainWindow::drawMultiClassROC(
    const std::vector<std::vector<std::pair<double, double>>>& curves
    ) {
    QPixmap pix(imgMultiRoc->size());
    pix.fill(QColor("#1a1a1a"));

    QPainter p(&pix);
    p.setRenderHint(QPainter::Antialiasing);

    int pad = 40;
    int w = pix.width() - 2 * pad;
    int h = pix.height() - 2 * pad;

    // Draw axes
    p.setPen(QPen(Qt::white, 2));
    p.drawLine(pad, pix.height() - pad, pix.width() - pad, pix.height() - pad);
    p.drawLine(pad, pad, pad, pix.height() - pad);

    // Axis labels
    p.drawText(pix.width() / 2 - 10, pix.height() - 10, "FPR");

    p.translate(15, pix.height() / 2 + 10);
    p.rotate(-90);
    p.drawText(0, 0, "TPR");
    p.rotate(90);
    p.translate(-15, -(pix.height() / 2 + 10));

    // Predefined colors for each class
    QColor colors[] = {
        Qt::red, Qt::green, QColor("#1f77b4"), Qt::yellow,
        Qt::magenta, Qt::cyan, QColor("#ff7f0e"),
        QColor("#8c564b"), QColor("#bcbd22"), Qt::white
    };

    // Plot each class curve
    for (int c = 0; c < 10 && c < curves.size(); ++c) {
        p.setPen(QPen(colors[c], 2));

        for (size_t i = 1; i < curves[c].size(); ++i) {
            p.drawLine(
                pad + curves[c][i - 1].first * w,
                (pix.height() - pad) - curves[c][i - 1].second * h,
                pad + curves[c][i].first * w,
                (pix.height() - pad) - curves[c][i].second * h
                );
        }
    }

    imgMultiRoc->setPixmap(pix);
}

/**
 * Displays an OpenCV image inside a Qt QLabel.
 * Handles grayscale and RGB conversion automatically.
 */
void MainWindow::displayImageOnLabel(const cv::Mat& img, QLabel* label) {
    if (img.empty()) return;

    cv::Mat rgb;

    // Convert to RGB format if needed
    if (img.channels() == 1)
        cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
    else
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    // Convert to QImage and display
    QImage qimg(
        (const unsigned char*)(rgb.data),
        rgb.cols,
        rgb.rows,
        rgb.step,
        QImage::Format_RGB888
        );

    label->setPixmap(
        QPixmap::fromImage(qimg).scaled(
            label->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
            )
        );
}
