#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTabWidget>
#include <opencv2/opencv.hpp>
#include "FacePCA.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void handleLoadDetectionImage();
    void handleRunDetection();
    void handleTrainPCA();
    void handleRecognizeFace();
    void handleEvaluate();

private:
    void setupLayout();
    void setupDetectionTab(QWidget *tab);
    void setupRecognitionTab(QWidget *tab);
    void setupPerformanceTab(QWidget *tab);

    void displayImageOnLabel(const cv::Mat& img, QLabel* label);
    void drawGlobalROC(const std::vector<std::pair<double, double>>& points);
    void drawMultiClassROC(const std::vector<std::vector<std::pair<double, double>>>& all_curves);

    QTabWidget *tabs;

    QLabel *imgDetection;
    QPushButton *btnLoadDetImage, *btnRunDetection;
    cv::Mat currentDetectionImage;

    QLabel *imgInput, *imgResult, *lblStatus;
    QPushButton *btnTrain, *btnRecognize;

    QLabel *lblAccuracy, *imgGlobalRoc, *imgMultiRoc;
    QPushButton *btnEvaluate;

    FacePCA *engine;
    cv::CascadeClassifier face_cascade;
    bool isModelTrained;
};

#endif // MAINWINDOW_H
