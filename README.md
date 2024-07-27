# Training Documentation for Vision Transformer (ViT)
## 1. Overview
**Model Name:** Vision Transformer (ViT) <br>
**Task:** Image Classification <br>
**Environment:**
* Python Version: 3.12.2
* PyTorch Version: 2.4.0+cu121
* PyTorch Lightning Version: 2.3.3
* Other relevant packages: Listed in requirements.txt

## 2. Model Architecture
**Model Type:** ViTForImageClassification <br>
**Number of Parameters:**
* Trainable Parameters: 85.8 M
* Total Parameters: 85.8 M
* Total Estimated Model Params Size: 343.225 MB
**Mode:**
* Model: Eval
* Loss Function: Train
* Loss Function Used: CrossEntropyLoss

## 3. Data
**Dataset Used:** cifar-10-batches-py (located at /home/intern_2024_pds/brian/lab/ViT/data) <br>
**Data Augmentation Techniques:**
* Conversion to PIL images
* Resizing and normalization using ViTImageProcessor:
```bash
from PIL import Image
import numpy as np
import torch

def convert_to_pil(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return Image.fromarray(image)

def preprocess_data(dataset, feature_extractor):
    inputs = [feature_extractor(images=convert_to_pil(img), return_tensors="pt") for img, _ in dataset]
    return inputs
```
**Batch Size:** 8

## 4. Training Configuration and Hardware
* Number of Epochs: 20
* Learning Rate: 5e-5
* Optimizer: AdamW
* Scheduler: ReduceLROnPlateau
* GPU/TPU: 3 GPUs (Tesla V100-PCIE-32GB and Tesla V100S-PCIE-32GB)
* RAM: 1.5 TiB

## 5. Training Logs
Epoch Details: Provide an example of the logs from an epoch.
```bash
Epoch 0: 100%|██████████| 1667/1667 [07:07<00:00,  3.90it/s, v_num=1, train_loss=0.610, train_acc=0.857]
```
## 6. Results
Log Directory: /home/intern_2024_pds/brian/lab/ViT/results/ViT/version_1
