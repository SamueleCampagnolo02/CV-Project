# CV-Project
# Syringe Object Detection System


## 1. Project Overview

Experiments began using the public TACO (Trash Annotations in Context) benchmark. The extreme class imbalance, sparse syringe instances, and background clutter made fine boundary detection unreliable for needles and plastic transparent syringe.
Transition to Dedicated Syringe Dataset: The project scope was rescoped to a single target class (syringe). Training on focused imagery enabled the model to capture the distinct geometry of transparent cylinders, plungers, scale markings, and needles.

Production Deployment: The finalized model was exported to ONNX format and integrated into a standalone C++ application using ONNX Runtime and OpenCV.

## 2. Software and Tools
- Training Frameworks & Libraries

Ultralytics (YOLOv8 / YOLO26): Core object detection framework and training pipeline

PyTorch: Underlying deep learning backend for model training and optimization

PyYAML: Parsing and generation of dataset configuration files (data.yaml)

C++ Inference Engine

OpenCV (DNN Module): ONNX model loading, image preprocessing (letterbox/CHW layout), forward pass inference, and Non-Maximum Suppression (NMS)

CMake: Cross-platform build automation and dependency configuration

Visual Studio 2026 Community (MSVC): C++17 native compiler toolchain

- Model Format

ONNX (Open Neural Network Exchange): Interoperable model serialization format exported from Ultralytics for native OpenCV DNN inference

- Data Annotation Tools

MakeSense.ai: Manual bounding box annotation and labeling for custom syringe imagery

- Datasets & Data Sourcing

TACO (Trash Annotations in Context): Public benchmark dataset for initial multi-class exploration

Open Images v7 (via FiftyOne): Sourced syringe-specific images to expand visual diversity

Pixabay & Pexels: Royalty-free repositories used for background and real-world clutter acquisition

Google Images: Targeted search filtered by Creative Commons licenses for domain-specific variations

- Cloud & Training Infrastructure

Kaggle: Hosted cloud platform utilizing NVIDIA Tesla T4 GPUs for training runs

Google Drive: Storage backend for cloud-to-local asset synchronization via gdown

- Local Development Environment

Python (Isolated taco-env venv): Scripting environment for data preparation and pipeline tooling

pip: Python package manager

PowerShell: Native Windows execution and automation shell

Git / GitHub: Distributed version control and project hosting

Visual Studio Code: Primary code editor and workspace environment
## 3. Pipeline Architecture

[ Input: Camera Stream / Video File / Images ]
                       │
                       ▼
[ Pre-processing (C++ / OpenCV) ]
  - Aspect-ratio preserving Letterbox (640x640, pad color: 114)
  - Color conversion: BGR -> RGB
  - Tensor Normalization: float32, range [0.0, 1.0] (CHW layout)
                       │
                       ▼
[ Inference Engine (ONNX Runtime) ]
  - Model: best.onnx (FP32, opset 20)
  - Execution Provider: CPU / CUDA
                       │
                       ▼
[ Post-processing ]
  - Confidence Thresholding (score >= 0.35)
  - Non-Maximum Suppression (NMS IoU >= 0.45)
  - Coordinate re-scaling to native resolution
                       │
                       ▼
[ Visual Output / Detection Coordinates ]


## 4. Repository Layout
CV-Project/
├─ src/
│   └──main.cpp                         # Core detection and visualization loop
├── sample_images/
├── build                               # Compiled binaries
├──  CMakeLists.txt                            
├── taco_prep/                          # COCO to YOLO annotation converters
│   ├── coco_to_yolo.py                 # Full TACO category converter
│   └── coco_to_yolo_syringe.py         # Filtered converter isolating syringe instances
│
├── taco_full_classes/                  # Phase 1: Multi-class TACO training & eval
│   ├── train_taco.ipynb                # Training notebook on TACO benchmark
│   ├── model_50epochs/                 # Checkpoints & curves (50 epochs)
│   ├── model_150epochs/                # Checkpoints & curves (150 epochs)
│   ├── evaluation/                     # Metric plots, PR curves, and matrices
│   ├── src/                            # C++ inference prototype for TACO
│   └── CMakeLists.txt                  # Build configuration
│
├──  syringe-detection/                  # Phase 2: Dedicated syringe detector
       ├── first-dataset/                  # Initial baseline single-class dataset & runs
       ├── extended-dataset/               # Extended dataset 
       │       ├── 500-epochs-training/        # Long-run baseline (640px,  500 epochs)
       │       ├── yolo-26n/  
       │       │     ├──  first-try-50-epochs          # 50  epochs training
       │       │     ├──  150-epocs-revised-parameters # 150 epochs training 
       │       └── yolo-26s/                   # small model
       │
       └── time-evaluation/                
             ├──CMakeLists.txt              # CMake configuration 
             └──src/
                  └── time-evaluation.cpp     # High-precision latency benchmark per frame             

## 5. Build and Run (C++)
Requirements
CMake >= 3.16
C++17 compatible compiler
OpenCV >= 4.5
ONNX Runtime >= 1.14

## WINDOWS


- Syringe detector:
mkdir build
cd build
cmake -DOpenCV_DIR="<path_to_your_opencv_build>/x64/vc16/lib" ..  (e.g.  cmake -DOpenCV_DIR="C:/dev/opencv/build/x64/vc16/lib" ..   )
cmake --build . --config Release

Execution
Syringe detector:
.\Release\syringe_detector.exe ..\..\extended-dataset\500-epochs-training\weights\best.onnx ..\test_project



- Time evaluation:
cd syringe-detection\time-evaluation
mkdir build
cd build
cmake -DOpenCV_DIR="<path_to_your_opencv_build>\x64\vc16\lib" ..   (e.g.  cmake -DOpenCV_DIR="C:/dev/opencv/build/x64/vc16/lib" ..   )
cmake --build . --config Release
 
.\Release\time_evaluation.exe ..\..\extended-dataset\500-epochs-training\weights\best.onnx ..\..\..\sample_images


CANCEL CONTENT INSIDE BUILD
if ((Get-Item .).Name -eq 'build') { Remove-Item * -Recurse -Force; "Build directory contents deleted." } else { Write-Error "WARNING: Current working directory is not 'build'!" }

DATASET
https://drive.google.com/file/d/1xsx4JIkyj-Akhs5JtXaWkQLjETJQeSaF/view?usp=sharing


## 6. Experimental Results

### TACO Waste Classification

| Experiment       | Classes | Epochs             | Precision | Recall | mAP50 | mAP50-95 |
|:-----------------|:-------:|:------------------:|:---------:|:------:|:-----:|:--------:|
| Full classes     | 60      | 50                 | 0.358     | 0.108  | 0.101 | 0.083    |
| Full classes     | 60      | 150 (incremental)  | 0.293     | 0.141  | 0.126 | 0.098    |
| Macro-categories | 7       | 50                 | 0.522     | 0.216  | 0.172 | 0.124    |

### Syringe Detection

| Experiment                              | Model   | Images | Epochs                   | Precision | Recall | mAP50 | mAP50-95 |
|:----------------------------------------|:-------:|:------:|:------------------------:|:---------:|:------:|:-----:|:--------:|
| 1. Baseline                             | YOLO26n | 92     | 50                       | 0.454     | 0.429  | 0.416 | 0.227    |
| 2. Extended, aggressive augmentation    | YOLO26n | 380    | 50                       | 0.519     | 0.396  | 0.394 | 0.151    |
| 3. Extended, revised parameters         | YOLO26n | 380    | 150                      | 0.627     | 0.468  | 0.493 | 0.234    |
| 4. YOLO26s comparison                   | YOLO26s | 380    | 150                      | 0.604     | 0.486  | 0.510 | 0.230    |
| 5. Extended + Open Images (best)        | YOLO26n | 474    | 385 (early stop @285)    | 0.763     | 0.485  | 0.558 | 0.260    |

*Baseline condition*: Experiment 1 (YOLO26n fine-tuned on 92 hand-annotated images) serves as the reference baseline against which all subsequent improvements are measured.

*Ground truth*: bounding box annotations created manually with MakeSense.ai for all custom syringe images; COCO-format annotations from TACO and Open Images v7 used as-is for their respective sources.

*Key findings*:
- Grouping TACO's 60 imbalanced categories into 7 macro-categories improved mAP50 by ~70%.
- Training from scratch with moderate augmentation outperformed fine-tuning from a checkpoint with aggressive augmentation (experiment 2 vs. 3).
- YOLO26n outperformed the larger YOLO26s on identical data (experiment 3 vs. 4), suggesting model capacity was not the limiting factor.
- Dataset size remained the strongest lever throughout: extending from 92 to 474 images (experiments 1 to 5) nearly doubled mAP50.

---

## 7. AI and Third-Party Code Declaration

### AI/LLM Tools

Claude (Anthropic) was used throughout this project as a development assistant, specifically for:
- Drafting and debugging Python scripts (dataset conversion, deduplication, visualization utilities)
- Drafting and debugging C++ inference code (OpenCV DNN pipeline, preprocessing, NMS)
- Assisting with Kaggle notebook structure for model training

All design decisions (dataset composition, model architecture choices, training strategy, evaluation methodology) were made by the project authors. AI-generated code was reviewed, tested, and adapted before inclusion.

### Third-Party Code and Libraries

| Library / Tool              | License      |
|:----------------------------|:-------------|
| Ultralytics (YOLO26, YOLOv8)| AGPL-3.0     |
| PyTorch                     | BSD-3-Clause |
| OpenCV                      | Apache 2.0   |
| ONNX                        | Apache 2.0   |
| CMake                       | BSD-3-Clause |
| FiftyOne                    | Apache 2.0   |
| PyYAML                      | MIT          |

---

## 8. References and Existing Implementations

### Datasets
- Proença, P. F., & Simões, P. (2020). TACO: Trash Annotations in Context for Litter Detection. [arXiv:2003.06975](https://arxiv.org/abs/2003.06975) — [github.com/pedropro/TACO](https://github.com/pedropro/TACO)
- Kuznetsova, A., et al. (2020). The Open Images Dataset V4. International Journal of Computer Vision. [storage.googleapis.com/openimages](https://storage.googleapis.com/openimages/web/index.html)

### Models
- Ultralytics YOLO26 / YOLOv8 documentation: [docs.ultralytics.com](https://docs.ultralytics.com)

### Relation to Existing Work
This project builds directly on the public TACO dataset and its baseline category structure, and on Ultralytics' pre-trained YOLO weights (transfer learning starting point).

---

## 9. Inputs and Outputs

### Input
- A single image, a folder of images, or a video/camera stream (frame-by-frame)
- Accepted image formats: `.jpg`, `.jpeg`, `.png`, `.webp`
- No fixed resolution required — images are resized internally via letterbox to 640×640 before inference

### Output
- For each detected object: a bounding box (pixel coordinates in the original image), a class label, and a confidence score
- Console output: per-image detection count and inference time (`time-evaluation` build)
- Visual output: input image annotated with bounding boxes and labels (`syringe_detector` build)

### Limitations
- The system detects a single class (syringe) in the dedicated detector; the TACO-based detector covers general waste categories but was not carried forward into the final C++ pipeline
- Detection performance degrades on images whose visual context differs substantially from the training distribution (e.g. indoor surfaces vs. the outdoor/waste scenes dominant in training)
- Not validated against a physical conveyor belt; timing-based speed estimates are theoretical
- Recall (0.485 on the best model) indicates roughly half of true syringes in a frame may go undetected — not suitable as a sole safety mechanism without human oversight