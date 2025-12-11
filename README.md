# Vietnamese Traffic Sign Recognition (ViT vs YOLOv8)

[![Open In Colab – ViT](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/ViT_TrafficSign.ipynb)
[![Open In Colab – YOLOv8](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/YOLOv8_TrafficSign.ipynb)

**VN**: Dự án nhận diện biển báo giao thông Việt Nam, so sánh **Vision Transformer (ViT)** và **YOLOv8-classification**. Repo cung cấp notebook huấn luyện – đánh giá, biểu đồ Loss/Accuracy, Confusion Matrix, và demo Streamlit.

**EN**: Vietnamese traffic sign classification with **ViT** and **YOLOv8-cls**. This repo includes training/evaluation notebooks, Loss/Accuracy curves, confusion matrices, and a Streamlit demo.

## 1. Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## 2. Data
- Download `myData.zip` from your storage (Google Drive, etc.).
- Unzip to `data/myData/`.
- Notebooks contain 80/10/10 split logic; YOLO uses `configs/traffic.yaml`.

## 3. Notebooks
- `notebooks/ViT_TrafficSign.ipynb` – ViT training & evaluation (Train/Val Loss & Acc, Test metrics).
- `notebooks/YOLOv8_TrafficSign.ipynb` – YOLOv8 classification training, per-epoch logging, 4-metric plots.

> Notebooks are kept lightweight (outputs cleared). Consider `nbstripout` for clean diffs.

## 4. Results (samples)
Images like:
- `results/vit_train_val_curves.png`
- `results/metrics_train_val_4curves.png`

## 5. Reproducibility
- Set random seeds in notebooks.
- Document hardware/software (GPU/CPU, RAM, CUDA, Torch).
- Publish trained weights via **GitHub Releases** (recommended) or Git LFS.

## 6. Streamlit (optional)
```bash
streamlit run app/app_streamlit_vit.py
# or
streamlit run app/app_streamlit_yolo.py
```

## 7. License
MIT – see `LICENSE`.

## 8. Citation
See `CITATION.cff`.
