# 🐦 Single-View 3D Bird Reconstruction

### Overview
This repository implements a **single-view 3D reconstruction framework for birds**. Given one image, the system predicts semantic keypoints, regresses pose/shape/camera parameters, and drives a parametric bird model with **Linear Blend Skinning (LBS)** to generate a controllable 3D mesh.

**Key modules**
- **Keypoint Detector**: predicts K semantic keypoints with visibility-aware loss.
- **Regressor**: takes normalized keypoints (+ optional mask features) to regress **25 joint rotations (6D)**, **bone-length shape scalars**, and **camera**.
- **Parametric Bird Model**: skeleton + mesh + skinning weights + forward kinematics; projects to 2D via least-squares weak projection.
- **Training/Eval**: MPJPE, PCK@α; ablation on heatmap vs coordinate regression.


---

### Features
- One-image → full **3D bird mesh** (topology-preserving, animation-ready).
- **Coordinate regression (FPN+SE)** alternative to heatmaps; stronger global-shape consistency.
- Reproducible **preprocessing**: bbox-guided segmentation (SAM), crop & resize to **256×256**, keypoint normalization to **[0,1]**.
- Clear **metrics**: MPJPE, PCK@0.05, PCK@0.10 with visualization helpers.

---


### Installation
**Requirements**
- Python 3.9+
- PyTorch 1.12+ (CUDA recommended)
- torchvision, numpy, scipy, opencv-python, matplotlib, pyyaml, tqdm
- (Optional) **Segment Anything (SAM)** for mask generation


### Data Preparation
1) **Download CUB-200-2011** separately (place under `data/CUB-200-2011`).  
2) **Preprocess** (bbox-guided SAM mask → crop/resize to 256×256; remap keypoints 15→14; normalize to [0,1]).
```
```

### Training
**Keypoint Detector**
```bash
keypoint_detector_train.ipynb
```
**Regressor**
```bash
regressor_train.ipynb
```

---

### Evaluation
```bash
python eval/evaluate.py --checkpoint checkpoints/regressor_best.pth
```


### Limitations & Future Work
- Sensitivity to domain shift (illumination/species/extreme viewpoints).  
- Coordinate regression lacks explicit spatial uncertainty under heavy occlusion.  
- Dependence on accurate preprocessing (crop/mask).  
**Future**: probabilistic regression (uncertainty), stronger structural priors (skeleton/graph constraints), domain generalization & TTA.

---

### Acknowledgements
- CUB-200-2011: dataset with 200 bird species, bbox, 15 keypoints.  
- Segment Anything (SAM): used for segmentation masks.  
- Community implementations of 6D rotation → SO(3) and LBS/FK routines.


