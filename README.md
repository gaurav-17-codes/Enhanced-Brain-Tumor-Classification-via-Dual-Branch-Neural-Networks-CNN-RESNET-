## Enhanced Brain Tumor Classification via Dual-Branch Neural Networks 

### Project Overview

This repository contains the complete implementation of a Dual-Branch Adaptive Fusion Neural Network designed to classify brain tumors (Glioma, Meningioma, Pituitary, and No Tumor) from MRI scans.

Standard deep learning baselines (like standalone ResNet50) suffer from algorithmic bias against rare tumors and catastrophic forgetting during transfer learning. This project solves these clinical bottlenecks by implementing a dynamically weighted loss function, a learnable fusion gate, and a mathematically calibrated label smoothing pipeline.

Final Diagnostic Efficacy: Macro F1-Score of 0.95 and near-perfect AUC metrics (0.98 - 1.00) across 7,200+ holdout images.


### Core Architectural Innovations

1. Dual-Branch Framework

Instead of relying on a single architecture, this model utilizes a "Two-Brain" approach:

Branch 1 (Custom CNN): Acts as a microscope, extracting fine-grained, localized pathological textures (crucial for distinguishing morphologically similar boundaries).

Branch 2 (ResNet50 Base): Acts as a macroscopic lens, capturing the global anatomical structure of the brain utilizing ImageNet pre-trained weights.

2. Dynamic Adaptive Fusion Gate ($\alpha$)

Standard multi-branch models use simple concatenation, which leads to "feature flooding" (where massive ResNet features drown out delicate CNN textures).

The Solution: We engineered a learnable Sigmoid gate ($\alpha$) that dynamically balances the feature branches. For every single MRI scan, the model autonomously shifts its mathematical focus, prioritizing the CNN for tricky textural boundaries and ResNet for global shape context.

Equation: $F_{fused} = \alpha \cdot F_{cnn} + (1 - \alpha) \cdot F_{resnet}$

3. Dynamically Weighted Loss Function ($w_i$)

To eradicate algorithmic bias caused by class imbalance, we replaced standard Categorical Cross-Entropy with a dynamically scaled gradient penalty. By multiplying the error gradient of minority classes by a high weight ($w_i$), the optimizer is physically forced to take massive corrective steps when it misclassifies rare tumors, ensuring statistically fair learning.

4. Label Smoothing Calibration ($\epsilon = 0.1$)

To prevent the model from pushing raw logits to infinity (which causes overfitting to background MRI noise), we implemented a label smoothing factor of 0.1. This lowers the target confidence boundary to 0.90, allowing the model to converge safely without becoming overconfident on noisy pixels.


### Clinical Results & Efficacy

The framework was evaluated on an aggregated dataset of 7,200+ MRI scans (fusing Figshare, SARTAJ, and BR35H datasets to ensure cross-domain robustness).

Key Metrics:
Macro Average F1-Score: 0.95
Glioma Precision: 1.00 (Zero False Positives: When the architecture flags a Glioma, the diagnosis is 100% certain).
Healthy Tissue Recall: ~1.00 (Virtually zero missed diagnoses for non-tumor cases).
Threshold Independence (ROC/AUC Analysis)

The model maintains exceptional discriminative power across all possible clinical confidence thresholds:

No Tumor AUC: 1.00
Pituitary AUC: 1.00
Meningioma AUC: 0.99
Glioma AUC: 0.98


### Getting Started

Prerequisites
Python 3.8+
TensorFlow 2.x
Jupyter Notebook
Installation & Execution
Clone the repository:
git clone [https://github.com/your-username/dual-branch-tumor-classification.git](https://github.com/your-username/dual-branch-tumor-classification.git)
cd dual-branch-tumor-classification

Install required dependencies:
pip install -r requirements.txt

Run the Notebook:
Open Brain_Tumor_Classification_Dual_Branch.ipynb in Jupyter Notebook or directly import it into Kaggle / Google Colab. The notebook contains the full end-to-end pipeline including data preprocessing, the progressive 4-stage training cycle, and evaluation metric generation.


### Future Scope: Full-Stack Clinical Deployment

The next phase of this project involves wrapping this frozen .h5 inference engine into a production-ready application:
Backend: Deploying the TensorFlow model as a lightweight Python FastAPI microservice.
Frontend: Building a secure Next.js application with NextAuth Role-Based Access Control (RBAC), allowing isolated dashboards for Doctors (inference triggers) and Administrators (analytics).

### Acknowledgments
A special thanks to our project guides, Dr. Richik Kashyap and Nirupam Sir (Assam University, Silchar), for their invaluable technical guidance, continuous support, and mentorship throughout the development of this research project.
