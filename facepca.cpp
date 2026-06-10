#include "FacePCA.h"
#include <iostream>

FacePCA::FacePCA(int components) : num_components(components) {}

/**
 * Detects a face in the image and preprocesses it for PCA.
 * Steps:
 * 1. Load image in grayscale
 * 2. Detect face using Haar Cascade
 * 3. Crop face region (or fallback to full image)
 * 4. Resize to fixed size (50x50)
 * 5. Flatten and convert to float
 */
cv::Mat FacePCA::detectAndPreprocess(const std::string& imagePath, cv::CascadeClassifier& cascade) {
    // Load image in grayscale
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) return cv::Mat();

    // Detect faces
    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(img, faces, 1.1, 4);

    cv::Mat face_roi;

    // If a face is detected, crop the first detected face
    if (!faces.empty()) {
        face_roi = img(faces[0]);
    }
    // Otherwise, use the entire image as fallback
    else {
        face_roi = img;
    }

    // Resize to a fixed size
    cv::Mat resized_face;
    cv::resize(face_roi, resized_face, cv::Size(50, 50));

    // Flatten the image into a single row vector
    cv::Mat flattened = resized_face.reshape(1, 1);

    // Convert pixel values to float for PCA computations
    cv::Mat float_face;
    flattened.convertTo(float_face, CV_32FC1);

    return float_face;
}

/**
 * Trains the PCA model using the provided training data.
 * Steps:
 * 1. Compute mean face
 * 2. Center the data
 * 3. Compute covariance matrix (efficient trick)
 * 4. Extract eigenvalues and eigenvectors
 * 5. Compute actual eigenfaces
 * 6. Normalize and keep top K components
 * 7. Project training data to PCA space
 */
void FacePCA::train(const cv::Mat& training_data, const std::vector<int>& labels) {
    training_labels = labels;
    int num_images = training_data.rows;

    // 1. Compute mean face
    cv::reduce(training_data, mean_face, 0, cv::REDUCE_AVG);

    // 2. Subtract mean from all images (A = X - mean)
    cv::Mat A = cv::Mat::zeros(training_data.size(), training_data.type());
    for (int i = 0; i < num_images; i++) {
        A.row(i) = training_data.row(i) - mean_face;
    }

    // 3. Compute covariance matrix using trick (L = A * A^T)
    cv::Mat L = A * A.t();

    // 4. Compute eigenvalues and eigenvectors
    cv::Mat evalues, evectors_L;
    cv::eigen(L, evalues, evectors_L);

    // 5. Compute actual eigenfaces (V = A^T * eigenvectors^T)
    cv::Mat actual_eigenfaces = A.t() * evectors_L.t();

    // 6. Normalize and keep top K eigenfaces
    eigenfaces = cv::Mat::zeros(actual_eigenfaces.rows, num_components, CV_32FC1);
    for (int i = 0; i < num_components; i++) {
        cv::Mat col = actual_eigenfaces.col(i);
        cv::normalize(col, col);
        col.copyTo(eigenfaces.col(i));
    }

    // 7. Compute weights for training images
    training_weights = project(training_data);
}

/**
 * Projects input faces into PCA (eigenface) space.
 * W = (X - mean) * eigenfaces
 */
cv::Mat FacePCA::project(const cv::Mat& faces) {
    cv::Mat A = cv::Mat::zeros(faces.size(), faces.type());

    // Center data by subtracting mean
    for (int i = 0; i < faces.rows; i++) {
        A.row(i) = faces.row(i) - mean_face;
    }

    // Project onto eigenface space
    return A * eigenfaces;
}

/**
 * Predicts the label of a test face using nearest neighbor.
 * Uses Euclidean distance between projected weights.
 */
int FacePCA::predict(const cv::Mat& test_face_weight, double& out_distance) {
    double min_dist = DBL_MAX;
    int best_label = -1;

    // Compare with all training weights
    for (int i = 0; i < training_weights.rows; i++) {
        double dist = cv::norm(training_weights.row(i), test_face_weight, cv::NORM_L2);

        if (dist < min_dist) {
            min_dist = dist;
            best_label = training_labels[i];
        }
    }

    out_distance = min_dist;
    return best_label;
}
