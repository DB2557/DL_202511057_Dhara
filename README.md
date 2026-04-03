# Deep Learning Assignment – Object Detection (R-CNN Family & YOLO)

## Student Information

**Name:** Dhara Bambhroliya<br>
**Student ID:** 202511057

---

## Assignment Overview

This repository contains implementations and analysis of classical and modern object detection techniques including:

* IoU (Intersection over Union) Demonstration
* Manual R-CNN Crop Loop
* Fast R-CNN with RoI Pooling
* Faster R-CNN Inference
* Non-Maximum Suppression (NMS) Algorithm
* YOLO Training Pipeline

The goal of this assignment is to understand the evolution of object detection models from R-CNN to YOLO and compare their efficiency and performance.

---



## Tasks Implemented

### Task 1 — IoU Demonstration

* Implemented IoU calculation
* Visual comparison of bounding boxes
* Mathematical explanation included

### Task 2 — Manual R-CNN Crop Loop

* Region proposal generation
* Image cropping
* Classification loop
* Visualization of proposals

### Task 3 — Fast R-CNN (RoI Pooling)

* Feature map extraction
* RoI pooling implementation
* Classification + bounding box regression
* **Execution time printed in output cells**

### Task 4 — Faster R-CNN Inference

* Pretrained Faster R-CNN model
* Object detection inference
* Bounding box visualization
* **Execution time printed in output cells**

### Task 5 — Non-Maximum Suppression (NMS)

* Manual NMS implementation
* IoU-based filtering
* Confidence thresholding

### Task 6 — YOLO Training

* Dataset loading (.jpg + .xml annotations)
* Annotation conversion
* YOLO model training
* Detection results visualization

---

## Conceptual Analysis

All conceptual questions are answered within the notebook using **Markdown cells** placed after each corresponding task.

Topics covered:

* R-CNN vs Fast R-CNN vs Faster R-CNN
* RoI Pooling explanation
* Role of Region Proposal Network
* YOLO advantages and limitations
* NMS importance
* IoU significance

---

## Execution Time Requirement

Execution times are printed for:

* Task 3 — Fast R-CNN
* Task 4 — Faster R-CNN

These are clearly visible in the notebook output cells.

---

## Requirements

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install numpy
pip install ultralytics
```

---

## How to Run

1. Clone repository

```bash
git clone <your-repo-link>
```

2. Open notebook

```bash
jupyter notebook lab4_DL_Assignment.ipynb
```

3. Run all cells sequentially

---



