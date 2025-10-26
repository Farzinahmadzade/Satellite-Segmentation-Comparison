# Satellite-Segmentation-Comparison

A comprehensive benchmark for comparing different deep learning models on satellite image segmentation tasks using the OpenEarthMap dataset. This project provides a flexible pipeline to train, evaluate, and visualize multiple state-of-the-art segmentation models.

# Overview

This repository aims to provide a clean and reproducible framework for benchmarking semantic segmentation models on satellite imagery. The focus is on:

1. Comparing multiple modern segmentation architectures.

2. Handling a real-world dataset (OpenEarthMap) with potential missing labels or images.

3. Providing visualizations and metrics for quick assessment.

# Dataset

OpenEarthMap Dataset is used, organized as follows:

```
<region_name>/
├─ images/   ← Satellite images (.tif)
├─ labels/   ← Corresponding segmentation masks (.tif)
```

## Notes on Dataset Handling
1. Only paired images and labels are used; incomplete files are automatically skipped.
2. Optionally, training can be limited to a small number of samples for quick testing.
3. Top regions can be filtered to focus on selected geographical areas.

# Configuration

All hyperparameters are defined in config.py:

IMAGE_SIZE = 256 → Images are resized to 256x256 for training efficiency.

BATCH_SIZE = 2 → Small batch due to limited GPU memory.

EPOCHS = 5 → Low number for initial experiments or quick testing.

LEARNING_RATE = 1e-3 → Default learning rate for Adam optimizer.

NUM_CLASSES = 9 → Classes defined according to OpenEarthMap labels.

TOP_REGIONS_ONLY → Allows filtering to only specified top regions.

# Models

We use Segmentation Models PyTorch (smp)
 to access modern segmentation architectures easily. Currently supported models:

Unet

Unet++

FPN

PSPNet

DeepLabV3

DeepLabV3+

Linknet

MAnet

PAN

UPerNet

Segformer

DPT

## Why SMP?

Provides pre-built, tested, and widely-used segmentation architectures.

Easy to swap encoders and configure number of classes.

Compatible with PyTorch, ensuring GPU acceleration and reproducibility.

# Project Structure
```
Satellite-Segmentation-Comparison/
│
├─ config.py           # Hyperparameters & settings
├─ dataset.py          # Custom Dataset for OpenEarthMap
├─ model.py            # Model factory using SMP
├─ utils.py            # Metrics and visualization helpers
├─ train.py            # Training script for a single model or all models
├─ outputs/            # Saved models and TensorBoard logs
└─ README.md           # This file
```

# Training

## Single model :
```python train.py --data_dir "PATH_TO_DATASET" --model_name unet --encoder_name resnet34 --output_dir "./outputs"```

## Quick test with limited samples:
```python train.py --data_dir "PATH_TO_DATASET" --model_name unet --encoder_name resnet34 --output_dir "./outputs" --limit_samples 10```

## Training all models sequentially:
```python train.py --data_dir "PATH_TO_DATASET" --model_name all --encoder_name resnet34 --output_dir "./outputs"```

# TensorBoard
## View training progress:
```tensorboard --logdir=./outputs/logs```

Scalars: Loss, IoU, Dice, Accuracy

Images: Input | Ground Truth | Prediction

# Metrics 
Metrics are implemented in utils.py:

1. IoU (Intersection over Union) → Measures overlap between predicted and ground truth masks.
2. Dice Coefficient → Another overlap metric, sensitive to small objects.
3. Accuracy → Pixel-wise correctness.

Visualizations allow comparing input, ground truth, and prediction side by side for each epoch.

# Design Choices
###### Small epochs & batch size:
Designed for testing or GPU-limited environments. Can be increased for full training.

###### Paired dataset filtering:
Ensures training only on valid image-mask pairs to prevent runtime errors.

###### Top regions filtering:
Allows focusing experiments on key regions of interest.

###### SMP models:
Provides a wide variety of architectures with minimal setup and consistent interface.

###### TensorBoard logging:
Essential for monitoring training, diagnosing problems, and visual comparison.

# References

OpenEarthMap Dataset: [https://github.com/
..., etc.]

Segmentation Models PyTorch: [https://smp.readthedocs.io/
]