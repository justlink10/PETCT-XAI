# FDG PET/CT Colorectal Classifier — XAI Attribution Scripts

Scripts for explaining a deep-learning FDG PET/CT classifier (colorectal cancer)
using occlusion-based attribution methods, evaluated with the Area Over the
Perturbation Curve (AOPC).

## Contents

- **`data_process.py`** — Shared utility module: loading NIfTI images/segmentations,
  patching/sliding-window helpers, and class-balancing (oversampling) utilities.
- **`pet_ct_xai.py`** — Generates SUV-based and spatial occlusion attribution maps
  per anatomical structure (from a segmentation), combines them into a global
  attribution map, and scores each structure's contribution via AOPC.
- **`permutation_test.py`** — Computes AOPC for a set of scans given precomputed
  attribution maps, by progressively occluding the highlighted voxels and
  measuring the resulting change in the classifier's predicted probability.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

`pet_ct_xai.py` and `permutation_test.py` each define placeholder paths near the
top of the file (e.g. `PET_DIR`, `SEG_DIR`, `MODEL_WEIGHTS_PATH`). Update these to
point to your local PET scan, segmentation, and trained model weight directories
before running:

```bash
python pet_ct_xai.py
python permutation_test.py
```

Both scripts expect NIfTI (`.nii`) PET scans and segmentation masks, and a Keras
classifier saved as `.h5` weights.
