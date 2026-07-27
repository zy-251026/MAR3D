## MAR3D

Official implementation of **MAR3D**, an attention-based mid-level feature representation incorporated 3D residual neural network for brain age estimation from T1-weighted brain MRI.

> Paper: **An Attention-based Mid-level Feature Representation Incorporated 3D Residual Neural Network for Brain Age Estimation**  
> Authors: Yu Zhang, Jiaru Yang, Masaaki Omura, Zhenyu Lei, Hideyuki Hasegawa, and Shangce Gao  
> Code: https://github.com/zy-251026/MAR3D

## Overview

Brain age estimation aims to predict the biological age of the brain from neuroimaging data. The discrepancy between predicted brain age and chronological age can provide a useful imaging biomarker for abnormal brain aging and may support early risk assessment for neurodegenerative and metabolic disorders.

MAR3D is designed for 3D MRI-based brain age estimation. It combines a 3D ResNet backbone with an attention mechanism-based module, named **AM Block**, to refine mid-level feature representations. The AM Block uses max-pooling sampling, regional multi-head attention, and max-unpooling based spatial restoration to enhance cross-regional feature modeling while preserving the 3D structure of brain MRI.

<img width="473" height="212" alt="mar3d-34" src="https://github.com/user-attachments/assets/b54443bc-c403-483b-9ba5-e5054d6c1a38" />

## Highlights

- 3D voxel-based brain age estimation from T1-weighted MRI.
- 3D ResNet backbone with attention-based mid-level feature representation.
- AM Block for compact regional attention over 3D residual features.
- Evaluation on 7,680 MRI scans from six public datasets with ages ranging from 6 to 96 years.
- Best average MAE of **2.298 years** under the repeated-split experimental setting reported in the paper.
- Model definitions for MAR3D-10, MAR3D-18, MAR3D-34, MAR3D-50, and MAR3D-101.

## Data Preparation

Public MRI datasets are not redistributed in this repository. Please download the datasets from their official sources and follow their licenses and data-use agreements.

<img width="458" height="258" alt="preprocessing" src="https://github.com/user-attachments/assets/90f48b49-1848-4913-8b65-9a011ab8f906" />

The paper used six public datasets:

| Dataset | Samples | Age range | Mean age |
| --- | ---: | ---: | ---: |
| ADNI | 2692 | 55.0-93.0 | 76.92 |
| CoRR | 3200 | 6.0-88.0 | 26.31 |
| SALD | 494 | 19.0-80.0 | 45.18 |
| DLBS | 315 | 20.6-89.1 | 54.61 |
| IXI | 563 | 19.9-86.3 | 48.65 |
| OASIS-1 | 416 | 18.0-96.0 | 52.70 |
| Overall | 7680 | 6.0-96.0 | 49.28 |

Expected local data layout:

```text
MAR3D/
├── dataset/
│   ├── label/
│   │   └── label2.csv
│   ├── subject_001.nii.gz
│   ├── subject_002.nii.gz
│   └── ...
└── preprocessing/
    ├── resample/
    ├── skullstrip/
    ├── n4/
    ├── registration/
    ├── segmentation/
    └── normal/
```

The label file should contain at least the following columns:

```csv
name,age
subject_001,72.0
subject_002,45.5
```

The `name` field should match or be contained in the corresponding MRI filename.

Create required folders before preprocessing:

```bash
mkdir -p dataset/label
mkdir -p preprocessing/{resample,skullstrip,n4,registration,segmentation,normal}
mkdir -p data model/trained_model
```

## Preprocessing

The preprocessing pipeline follows four main steps:

1. Skull stripping
2. N4 bias field correction
3. Template registration to MNI152 space
4. Voxel intensity normalization

Example usage:

```python
import glob
from preprocessing import Processor

for path in glob.glob("./dataset/*.nii.gz"):
    Processor(path).start()
```

After preprocessing, normalized MRI volumes are expected under:

```text
preprocessing/normal/
```

In the paper experiments, each MRI volume was standardized to approximately `182 x 218 x 182` with isotropic spatial resolution of `1 mm^3`.

## Quick Model Check

You can instantiate MAR3D directly from `model/MAR3D.py`:

```python
import torch
from model.MAR3D import MAR3D_34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MAR3D_34().to(device)
model.eval()

x = torch.randn(1, 1, 182, 218, 182).to(device)

with torch.no_grad():
    pred_age = model(x)

print(pred_age.shape)  # torch.Size([1, 1])
```

## Training

`main.py` provides a reference training pipeline. The default experimental setting in the paper used Adam with learning rate `1e-3`, 50 epochs, and small batch sizes due to the high memory cost of 3D MRI.

Example:

```bash
python main.py --model_depth 34 --batch_size 1 --epoch 50 --lr 0.001
```

## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{zhang2026mar3d,
  title   = {An Attention-based Mid-level Feature Representation Incorporated 3D Residual Neural Network for Brain Age Estimation},
  author  = {Zhang, Yu and Yang, Jiaru and Omura, Masaaki and Lei, Zhenyu and Hasegawa, Hideyuki and Gao, Shangce},
  journal = {IEEE/CAA Journal of Automatica Sinica},
  year    = {2026},
  note    = {In press}
}
```
