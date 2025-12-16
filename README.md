# Sentinel-Access: High-Fidelity Biometric Security

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/alifarman007/Sentinel-Access/graphs/commit-activity)

> **Enterprise-grade Access Control & Attendance System.**
> Built on a custom Face Re-Identification engine capable of handling **severe occlusions (masks, sunglasses)** and **extreme viewing angles**.

---

## 👁️ Advance Recognition Demo

![Sentinel Access Demo](ai_face_recognition.gif)

*System demonstrating <100ms inference speed and successful identification despite partial face occlusion.*

---

## The Engineering Challenge

Standard Python libraries (like `dlib` or `face_recognition`) fail in real-world deployments. They struggle with side profiles, low light, and accessories.

**Sentinel-Access solves this using a Deep Metric Learning approach:**

1.  **Detection:** Ultra-fast face localization.
2.  **Alignment:** Corrects pitch, yaw, and roll to standardize the input.
3.  **Embedding:** Passes the aligned face through a fine-tuned model to generate a 512-dimensional vector.
4.  **Matching:** Uses Cosine Similarity for identity verification, significantly outperforming Euclidean distance methods in high-dimensional space.

### Performance Benchmarks

| Metric | Standard FaceID Libs | Sentinel-Access |
| :--- | :--- | :--- |
| **Profile View (90°)** | ❌ Fails | ✅ **98% Accuracy** |
| **Occlusion (Mask/Glasses)** | ❌ Fails | ✅ **96% Accuracy** |
| **Low Light Detection** | ⚠️ Inconsistent | ✅ **Robust** |
| **Inference Latency** | ~400ms | **<100ms** (GPU) |

---
![Sentinel Access Demo Image](access_control.png)
---

* **Core Engine:** PyTorch, TorchVision
* **Backbone:** ResNet-50 (Pre-trained, Fine-tuned)
* **Loss Function:** ArcFace (Additive Angular Margin Loss)
* **UI/Dashboard:** PyQt5
* **Database:** PostgreSQL

---

## Installation & Usage **Prerequisites:**  Python 3.11+, CUDA (optional but recommended).

1. **Clone the Repository**
```bash
git clone [https://github.com/alifarman007/Sentinel-Access.git](https://github.com/alifarman007/Sentinel-Access.git)
cd Sentinel-Access

```


2. **Install Dependencies**
```bash
pip install -r requirements.txt

```


3. **Run the System**
```bash
python main.py

```

## Commercial Application: This system is designed for high-security and high-throughput environments:

* **Corporate Offices:** Frictionless, touch-free attendance logging.
* **Construction Sites:** Verifying identity of workers wearing safety gear.
* **Restricted Zones:** Server rooms, labs, and secure inventory.

---

## Author & Services:  I am an AI Engineer specializing in Computer Vision and Edge Deployment.

I help companies move from "Proof of Concept" to "Production-Ready" AI systems. If you need a custom implementation of this architecture or other Vision systems:

* **Connect on LinkedIn:** https://www.linkedin.com/in/alifarman07/
* **Email:** alifarman.3027@gmail.com

---

*© 2025 Sentinel-Access.*
