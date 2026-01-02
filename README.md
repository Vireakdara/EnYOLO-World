# EnYOLO-World

**Enhanced YOLO-World with Improved YOLOv9 Backbone and FiLM-Driven PAN for Object Detection**

[![Paper](https://img.shields.io/badge/Paper-ELCVIA-blue)](https://elcvia.cvc.uab.cat/)
[![License](https://img.shields.io/badge/License-GPL--3.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

## Overview

EnYOLO-World is an enhanced object detection framework that addresses two persistent bottlenecks in YOLO-based architectures: progressive loss of spatial detail in deeper backbone stages, and rigid feature aggregation strategies that fail to adapt to varying scene complexity.

Our approach combines:
- **PGI-Enhanced GELAN Backbone**: Maintains fine-grained representations throughout the network hierarchy using Programmable Gradient Information modules
- **FiLM-Driven Path Aggregation Network (FiLM-PAN)**: Leverages semantic embeddings to perform channel-wise affine recalibration during multiscale feature fusion at linear computational complexity

![Main Architecture Design](https://github.com/user-attachments/assets/2c75175b-1ef0-4fa9-a54c-66abe3d67574)

## Key Results

On COCO val2017:

| Model | AP50:95 | AP50 | AP75 | APS | APM | APL | Params | FLOPs | FPS |
|-------|---------|------|------|-----|-----|-----|--------|-------|-----|
| YOLO-World-L | 51.0 | 68.3 | 55.8 | 33.2 | 55.4 | 66.5 | 23.4M | 112G | 35 |
| GLIP-T | 51.4 | 68.5 | 56.1 | 33.8 | 56.0 | 67.1 | 110.6M | 300G | 12 |
| **EnYOLO-World-L** | **52.2** | **69.4** | **57.7** | **34.5** | **56.8** | **68.0** | 26.1M | 132G | 31 |

EnYOLO-World-L achieves:
- **+1.2 AP** over YOLO-World-L with only 5.6% increase in FLOPs
- **+0.8 AP** over GLIP-T while using **4.2× fewer parameters**

## Installation

```bash
# Clone the repository
git clone https://github.com/Vireakdara/EnYOLO-World.git
cd EnYOLO-World

# Create conda environment
conda create -n enyolo python=3.8 -y
conda activate enyolo

# Install PyTorch (adjust CUDA version as needed)
pip install torch>=2.0.0 torchvision>=0.15.0

# Install dependencies
pip install -r requirements.txt

# Install MMYOLO and MMDetection
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
mim install "mmyolo"
```

## Quick Start

### Inference

```python
from enyolo_world import EnYOLOWorld

# Load pretrained model
model = EnYOLOWorld.from_pretrained('enyolo-world-l')

# Run inference
results = model.predict('path/to/image.jpg', texts=['person', 'car', 'dog'])

# Visualize results
model.visualize(results, save_path='output.jpg')
```

### Training

```bash
# Single GPU training
python train.py configs/enyolo_world_l_coco.py

# Multi-GPU training
bash dist_train.sh configs/enyolo_world_l_coco.py 4
```

### Evaluation

```bash
# Evaluate on COCO val2017
python eval.py configs/enyolo_world_l_coco.py weights/enyolo_world_l.pth
```

## Model Zoo

| Model | Backbone | AP50:95 | Weights | Config |
|-------|----------|---------|---------|--------|
| EnYOLO-World-S | imYOLOv9-S | 43.9 | [download]() | [config](configs/enyolo_world_s_coco.py) |
| EnYOLO-World-M | imYOLOv9-M | 49.2 | [download]() | [config](configs/enyolo_world_m_coco.py) |
| EnYOLO-World-L | imYOLOv9-L | 52.2 | [download]() | [config](configs/enyolo_world_l_coco.py) |

*Pretrained weights will be released upon paper acceptance.*

## Architecture Details

### FiLM-PAN Forward Pass

```
Input: Backbone features {C3, C4, C5}, Text embedding T
Output: Fused pyramid features {P3, P4, P5}

1. Generate FiLM parameters (γ, β) from text embedding T
2. Initial Text-to-Image Calibration: C̃_l = γ_l ⊙ C_l + β_l
3. Top-down pathway with 2T-CSPLayer fusion
4. Bottom-up pathway with 2T-CSPLayer fusion  
5. Final FiLM refinement on P3, P4, P5
```

### 2T-CSPLayer (Text-Guided CSPLayer)

The core fusion block applies Feature-wise Linear Modulation:

```
V'_{h,w,c} = γ^c_l · V_{h,w,c} + β^c_l
```

where γ controls channel importance and β shifts feature distributions for semantic alignment.

## Project Structure

```
EnYOLO-World/
├── configs/                 # Configuration files
├── enyolo_world/           
│   ├── models/             # Model definitions
│   │   ├── backbone/       # PGI-enhanced YOLOv9
│   │   ├── neck/           # FiLM-PAN
│   │   └── head/           # Detection heads
│   ├── datasets/           # Dataset utilities
│   └── utils/              # Helper functions
├── tools/                  # Training/evaluation scripts
├── weights/                # Pretrained weights
└── requirements.txt
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{dara2025enyoloworld,
  title={EnYOLO-World: Improved YOLOv9 and FiLM-Driven PAN for Object Detection},
  author={Dara, Ly Vireak and Wang, Rongfang and Guo, Jiaxuan},
  journal={Electronic Letters on Computer Vision and Image Analysis},
  year={2025}
}
```

## Acknowledgements

This work builds upon:
- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- [MMYOLO](https://github.com/open-mmlab/mmyolo)
- [MMDetection](https://github.com/open-mmlab/mmdetection)

This work was supported by the National Natural Science Foundation of China (No.62176196) and the Key Industry Innovation Chain Project of Shaanxi (No.2024NCZDCYL-05-04).

## License

This project is released under the [GPL-3.0 License](LICENSE).

## Contact

- **Ly Vireak Dara** - [GitHub](https://github.com/Vireakdara)
- **Corresponding Author**: Rongfang Wang (rfwang@xidian.edu.cn)

---

*School of Artificial Intelligence, Xidian University, Xi'an, Shaanxi, China*
