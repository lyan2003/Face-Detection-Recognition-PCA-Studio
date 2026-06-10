# Face Detection & Eigenfaces Recognition Studio

A cross-platform, high-performance C++ computer vision application engineered utilizing the Qt Framework and OpenCV core matrix representations. This toolkit delivers an end-to-end analytical pipeline for facial feature extraction, real-time cascade-based face detection, dimensionality reduction via Principal Component Analysis (PCA/Eigenfaces), and statistical performance evaluation utilizing Receiver Operating Characteristic (ROC) curve metrics.

---

## Technical Pipeline Architecture

The toolkit decouples intensive matrix operations—such as high-dimensional covariance matrix decomposition and eigenspace projections—from the graphical user interface. This guarantees seamless, real-time multi-frame rendering and continuous parameter sweeping.

```text
+-------------------------------------------------------------------------+
|                               QT GUI LAYER                              |
|    (Dynamic Image Canvases, Interactive Threshold Sliders, ROC Plots)   |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                     ALGORITHMIC TOOLKIT CORE (C++)                      |
|    [Haar Face Detector]   --->    [PCA / Eigenfaces Engine]            |
|   (Cascade Classifiers)          (Covariance, Eigenvectors, Distance)   |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           OPENCV MATRIX CORE                            |
|       (cv::Mat Matrices, Eigenspace Projections, ROC Analytics)         |
+-------------------------------------------------------------------------+

```

---

## Core Algorithmic Capabilities

### 1. Cascade-Based Face Detection Layout

* **Haar-like Feature Evaluation:** Integrates Viola-Jones cascade structures via `haarcascade_frontalface_default.xml` to isolate facial regions of interest (ROIs) across grayscale or color matrices.
* **Multi-Scale Search Window:** Dynamically handles scale variations and bounds facial bounding boxes efficiently before forwarding pixel matrices to the recognition core.

### 2. Dimensionality Reduction via Principal Component Analysis (PCA)

Transforms high-dimensional facial pixel spaces into a lower-dimensional subspace spanned by the most significant statistical variances (Eigenfaces):

* **Mean Face Computation:** Given a set of face vectors $\Gamma_1, \Gamma_2, \dots, \Gamma_M$, the algorithm builds a foundational mean image layout matrix:

$$\Psi = \frac{1}{M}\sum_{i=1}^{M} \Gamma_i$$

* **Facial Mean Subtraction:** Computes the unique deviation vector for each image structure:

$$\Phi_i = \Gamma_i - \Psi$$

* **Surrogate Covariance Decomposition:** Solves large-scale dimensionality issues by analyzing the surrogate matrix $L = A^T A$ (where $A = [\Phi_1, \Phi_2, \dots, \Phi_M]$), yielding lower-dimensional eigenvectors that map cleanly to the true covariance matrix space $\mathbf{C} = A A^T$.

### 3. Eigenspace Matching & Distance Metrics

* **Face Database Projection:** Projects test faces into the established "Face Space" to generate localized feature weight vectors.
* **Classification Subsystem:** Evaluates distance configurations (Euclidean or Mahalanobis metrics) between projected test weights and historical dataset templates to securely confirm subject identity.

### 4. Performance Analytics & ROC Curve Synthesis

* **Threshold Sweeping Optimization:** Systematically shifts matching distance acceptance bounds to calculate diagnostic rates across standard face datasets.
* **ROC Curve Vector Compilation:** Evaluates the trade-off characteristics by mapping the True Positive Rate (Sensitivity) directly against the False Positive Rate (1 - Specificity):

$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

---

## Application Output Gallery

### 1. Cascade-Based Face Detection Results
* **Facial Region of Interest (ROI) Extraction (Grayscale & Color):**
<img width="1502" height="913" alt="image" src="https://github.com/user-attachments/assets/98a196a9-4d5b-4cc4-8819-60a944a707d8" />

### 2. PCA Eigenfaces & Feature Subspaces
* **Calculated Mean Face & Top Shaded Eigenfaces:**
<img width="1502" height="913" alt="image" src="https://github.com/user-attachments/assets/5dfa5662-d433-47c2-9753-19d9a715daf6" />

### 3. Biometric Performance & Statistical Analytics
* **Dynamic Distance Evaluation and Live ROC Curve Plotting:**
<img width="1502" height="913" alt="image" src="https://github.com/user-attachments/assets/ff72d0ca-7495-417c-aefd-2ab219588158" />

---
## Repository Directory Tree

```text
Face-Detection-Recognition-PCA-Studio/
├── CMakeLists.txt                      # Master cross-platform build pipeline configurations
├── .gitignore                          # Excludes local compiler cache, .qtcreator configs, and build targets
├── haarcascade_frontalface_default.xml # Pre-trained Viola-Jones cascade model for facial detection
├── main.cpp                            # Application bootstrap and primary GUI thread initialization
├── mainwindow.cpp                      # Master GUI layout logic, canvas controls, and signal routing
├── mainwindow.h                        # Structural UI slots, view matrix allocations, and window actions
├── mainwindow.ui                       # Graphical XML layout detailing user dashboard and plot modules
├── facepca.cpp                         # Implementation of PCA matrix solvers, Eigenface computation, and ROC curves
├── facepca.h                           # Declarations and mathematical namespaces for matching logic
└── dataset/                            # Evaluation directory housing target face database files

```

---

## Toolchain Setup and Deployment

### Prerequisites

* **Build Environment:** CMake (Version 3.16 or higher).
* **Graphical Framework:** Qt Framework Core Modules (Qt5 / Qt6 Widgets distribution).
* **Matrix Dependency:** OpenCV Development Libraries (For `cv::Mat` structures and Cascade Classifiers).
* **Compiler Requirements:** C++17 compliant compiler environment (MinGW, GCC, or Clang).

### Build Pipeline

1. Clone the repository and navigate into the source root:

```bash
git clone [https://github.com/lyan2003/Face-Detection-Recognition-PCA-Studio.git](https://github.com/lyan2003/Face-Detection-Recognition-PCA-Studio.git)
cd Face-Detection-Recognition-PCA-Studio

```

2. Initialize and configure an isolated build tracking tree:

```bash
mkdir build && cd build
cmake ..

```

3. Trigger the native compiler engine to compile and link all targets:

```bash
cmake --build .

```

4. Run the biometric computer vision engine application:

```bash
./FacePCAStudioApp
