#ifndef FACEPCA_H
#define FACEPCA_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string> // مهمة جداً عشان std::string

class FacePCA {
public:
    FacePCA(int components = 50);

    cv::Mat detectAndPreprocess(const std::string& imagePath, cv::CascadeClassifier& cascade);

    void train(const cv::Mat& training_data, const std::vector<int>& labels);
    cv::Mat project(const cv::Mat& faces);
    int predict(const cv::Mat& test_face_weight, double& out_distance);

    cv::Mat getMeanFace() { return mean_face; }
    cv::Mat getEigenfaces() { return eigenfaces; }

private:
    int num_components;
    cv::Mat mean_face;
    cv::Mat eigenfaces;
    cv::Mat training_weights;
    std::vector<int> training_labels;
};

#endif
