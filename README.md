Enhanced Brain Tumor Classification via Dual-Branch Neural Networks

### Project Overview

This repository contains the complete implementation of a Dual-Branch Adaptive Fusion Neural Network designed to classify brain tumors (Glioma, Meningioma, Pituitary, and No Tumor) from MRI scans.

Standard deep learning baselines (like standalone ResNet50) suffer from algorithmic bias against rare tumors and catastrophic forgetting during transfer learning. This project solves these clinical bottlenecks by implementing a dynamically weighted loss function, a learnable fusion gate, and a mathematically calibrated label smoothing pipeline.

Final Diagnostic Efficacy: Macro F1-Score of 0.95 and near-perfect AUC metrics (0.98 - 1.00) across 7,200+ holdout images.
