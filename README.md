# SEMA-Optimised: Robust Underwater Classification 🐢🐠

[![Institution](https://img.shields.io/badge/Institution-IIT%20Goa-blue)](https://iitgoa.ac.in/)
[![Paper](https://img.shields.io/badge/Status-BTP%20Report-success)](./BTP_Report%20(3).pdf)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

**Optimization and Enhancement of Self-Expanding Models (SEMA) for Robust Continual Learning in Autonomous Underwater Vehicles (AUVs).**

## 🌊 The Challenge

Marine ecosystems are dynamic and non-stationary. Monitoring them using Autonomous Underwater Vehicles (AUVs) presents a unique **"Stability-Plasticity"** dilemma:
1.  **No Cloud Connectivity:** AUVs operate with near-zero bandwidth; uploading data for offline retraining is impossible.
2.  **Catastrophic Forgetting:** Traditional deep learning models suffer from "Instant Amnesia" when learning new tasks (e.g., moving from classifying Sea Turtles to Clownfish).
3.  **Resource Constraints:** AUVs have limited battery life and cannot run massive backpropagation on historical data.

## 🚀 Solution: The Enhanced SEMA Architecture

This project optimizes the baseline SEMA (Self-Expansion of Pre-trained Models) architecture. We moved from a naive expansion model to a **Memory-Free Hybrid Architecture** that achieves state-of-the-art retention without requiring an external memory buffer.

### Key Innovations

| Feature | Baseline SEMA Flaw | Our Optimized Solution |
| :--- | :--- | :--- |
| **Expansion Logic** | **Panic Expansion:** Fixed thresholds caused false alarms in noisy underwater data. | **Adaptive Thresholding:** Dynamic statistical threshold using Z-scores to filter noise from real distribution shifts. |
| **Routing** | **Dumb Routing:** Linear layers averaged experts, reducing precision. | **Attention-Based Routing:** Key-Query mechanism for content-aware expert selection. |
| **Knowledge Retention** | **Frozen Wall:** Rehearsal failed due to frozen backbones blocking gradients. | **Teacher-Student Distillation:** LWF (Learning Without Forgetting) to anchor the student model to the teacher's memory. |
| **Classification** | **Recency Bias:** Linear heads favored new classes, dropping old task accuracy. | **NCM Classifier:** Nearest Class Mean classifier based on geometric centroids and Euclidean distance. |

## 🧠 Methodology Overview

### 1. Adaptive Thresholding
Instead of a fixed scalar, we utilize a dynamic threshold $\tau$ derived from a running buffer of reconstruction error z-scores.
$$\tau_{dynamic} = \mu_{buffer} + k \cdot \sigma_{buffer}$$
This prevents the "infinite loop" expansion issue seen in the baseline.

### 2. Attention-Based Routing
We implemented a Query-Key attention mechanism:
* **Query (Q):** Input image feature map.
* **Key (K):** Down-projection weights of adapters.
This ensures an image with specific features (e.g., scales) aligns mathematically with the relevant expert (e.g., Fish Expert), ignoring irrelevant ones (e.g., Coral Expert).

### 3. Nearest Class Mean (NCM)
To eliminate recency bias, we replaced the trainable linear head with NCM. A test sample is assigned to the nearest class prototype (geometric centroid) regardless of when that class was learned.

## 📊 Results

The optimized system was evaluated on **CIFAR-100**, **ImageNet-R**, and **ImageNet-A**.

* **Retention (CIFAR-100):** Achieved **90.65%** average incremental accuracy, significantly outperforming the baseline.
* **Robustness (ImageNet-R):** Achieved **83.65%** on 5-task splits, demonstrating strong generalization to out-of-distribution data.
* **Efficiency:** Adaptive thresholding successfully eliminated "Panic Expansion," reducing training time per task significantly.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch
* Torchvision
* Timm (for ViT backbones)

### Setup
```bash
git clone [https://github.com/EUPHORIA-7/SEMA_Optimised.git](https://github.com/EUPHORIA-7/SEMA_Optimised.git)
cd SEMA_Optimised
pip install -r requirements.txt
