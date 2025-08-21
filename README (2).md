# 🐦 Single-View 3D Bird Reconstruction

### Overview
This repository implements a **single-view 3D reconstruction framework for birds**. Given one image, the system predicts semantic keypoints, regresses pose/shape/camera parameters, and drives a parametric bird model with **Linear Blend Skinning (LBS)** to generate a controllable 3D mesh.

**Key modules**
- **Keypoint Detector**: predicts K semantic keypoints with visibility-aware loss.
- **Regressor**: takes normalized keypoints (+ optional mask features) to regress **25 joint rotations (6D)**, **bone-length shape scalars**, and **camera**.
- **Parametric Bird Model**: skeleton + mesh + skinning weights + forward kinematics; projects to 2D via least-squares weak projection.
- **Training/Eval**: MPJPE, PCK@α; ablation on heatmap vs coordinate regression.

> This README mirrors the content of your paper and provides runnable entry points and configs.

---

### Features
- One-image → full **3D bird mesh** (topology-preserving, animation-ready).
- **Coordinate regression (FPN+SE)** alternative to heatmaps; stronger global-shape consistency.
- Reproducible **preprocessing**: bbox-guided segmentation (SAM), crop & resize to **256×256**, keypoint normalization to **[0,1]**.
- Clear **metrics**: MPJPE, PCK@0.05, PCK@0.10 with visualization helpers.

---

### Directory Layout
```
├─ data/
│  ├─ CUB-200-2011/          # raw dataset (downloaded separately)
│  └─ processed/             # preprocessed images/masks/keypoints (auto-generated)
│  
├─ models/
│  ├─ detector/              # ResNet-50 baseline & FPN+SE heads
│  ├─ regressor/             # MLP/CNN branches, 6D rotation → SO(3)
│  └─ bird_model/            # skeleton, skin weights, FK/LBS, projection
├─ scripts/
│  ├─ prepare_cub.py         # preprocess:  SAM masks, resize
│  └─ visualize.py           # draw kps/meshes; qualitative comparisons
├─ train/
│  ├─ train_detector.py
│  └─ train_regressor.py
├─ eval/
│  └─ evaluate.py

└─ README.md
```

> If some scripts are not in your repo yet, use these names when you add them to keep the README actionable.

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
```bash
python scripts/prepare_cub.py \
  --root data/CUB-200-2011 \
  --out  data/processed \
  --resize 256 \
  --kp-remap cub15_to_14.json \
  --use-sam
```
This script writes:
```
data/processed/
  images/*.png
  masks/*.png
  keypoints/*.json    # normalized [0,1] coords (+ visibility)
  splits/{train.txt,val.txt}
```

---

### Configs
`configs/detector.yaml`
```yaml
seed: 42
data:
  root: data/processed
  img_size: 256
  num_keypoints: 14
model:
  backbone: resnet50
  head: fpn_se          # options: heatmap, fpn_se
  heatmap_out: 64
train:
  batch_size: 32
  epochs: 160
  lr: 1.0e-3
  aug: {flip: true, affine: true, color_jitter: true}
loss:
  type: masked_l2       # for coordinate regression
```

`configs/regressor.yaml`
```yaml
seed: 42
data:
  root: data/processed
  img_size: 256
  num_keypoints: 14
  kp_subset: 12         # 12 points used downstream
model:
  pose_mlp: [36, 512, 512, 153]  # 25*6 + 3 camera
  shape_cnn: {in_size: 64, out_dim: 24} # bone-length scalars (root fixed to 1.0)
  rotation: sixd
  bird_model:
    joints: 25
    use_lbs: true
train:
  batch_size: 64
  epochs: 140
  lr: 1.0e-3
loss:
  reprojection: smooth_l1
  bone_prior:
    weight: 0.1
    mean_file: null     # or path-to-prior.npz
```

---

### Training
**Keypoint Detector**
```bash
python train/train_detector.py --config configs/detector.yaml
```
**Regressor**
```bash
python train/train_regressor.py --config configs/regressor.yaml \
  --detector-ckpt checkpoints/detector_best.pth
```

---

### Evaluation
```bash
python eval/evaluate.py --checkpoint checkpoints/regressor_best.pth
```
**Metrics reported in the paper**
| Model                 | Supervision | MPJPE ↓ | PCK@0.05 ↑ | PCK@0.10 ↑ |
|----------------------|-------------|---------|------------|------------|
| Detector (baseline)  | Heatmap     | 41.524  | 0.133      | 0.362      |
| Detector (FPN+SE)    | CoordReg    | 39.816  | 0.207      | 0.487      |

> The substantial PCK gains indicate more predictions moved within tighter thresholds, while a few large residuals keep MPJPE modestly improved—consistent with stronger global-shape cues from FPN+SE.

---

### Reproducibility
- Fixed random seeds (config).  
- Input resolution fixed to **256×256**; synchronized keypoint normalization.  
- Visibility-masked losses; evaluation excludes occluded points by mask.  

---

### Limitations & Future Work
- Sensitivity to domain shift (illumination/species/extreme viewpoints).  
- Coordinate regression lacks explicit spatial uncertainty under heavy occlusion.  
- Dependence on accurate preprocessing (crop/mask).  
**Future**: probabilistic regression (uncertainty), stronger structural priors (skeleton/graph constraints), domain generalization & TTA.

---

### Citation
If you use this repository, please cite:
```bibtex
@article{yourpaper2025,
  title   = {Single-View 3D Bird Reconstruction via Deep Learning},
  author  = {Your Name},
  journal = {TBD},
  year    = {2025}
}
```

### License
MIT (unless you prefer another). CUB-200-2011 and SAM follow their respective licenses.

### Acknowledgements
- CUB-200-2011: dataset with 200 bird species, bbox, 15 keypoints.  
- Segment Anything (SAM): used for segmentation masks.  
- Community implementations of 6D rotation → SO(3) and LBS/FK routines.

### 训练 & 测试
```bash
# 训练关键点检测器
python train/train_detector.py --config configs/detector.yaml

# 训练参数回归器
python train/train_regressor.py --config configs/regressor.yaml \
  --detector-ckpt checkpoints/detector_best.pth

# 评估
python eval/evaluate.py --checkpoint checkpoints/regressor_best.pth
```

