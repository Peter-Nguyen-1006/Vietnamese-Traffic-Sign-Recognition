import os, time, re
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16

st.set_page_config(page_title="Traffic Sign - ViT Demo", layout="wide")
st.title("🚦 Vietnamese Traffic Sign – ViT Demo")

# Sidebar
ckpt_path = st.sidebar.text_input("ViT checkpoint (.pth)", value="outputs/best_vit.pth")
labels_csv = st.sidebar.text_input("labels.csv (IDClass,Name - optional)", value="")
imgsz = st.sidebar.number_input("Input size", min_value=64, max_value=640, value=224, step=32)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=True)

device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

def load_label_map_from_csv(path):
    if not path or not os.path.isfile(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Flexible column detection
    def norm(x): return re.sub(r"[^a-z0-9]+","", str(x).lower())
    cols = {c: norm(c) for c in df.columns}
    df = df.rename(columns=cols)
    id_col = next((c for c in ["idclass","classid","id","index","class"] if c in df.columns), None)
    nm_col = next((c for c in ["name","label","classname","ten","signname","viname"] if c in df.columns), None)
    if id_col is None or nm_col is None:
        return None
    df = df[[id_col, nm_col]].dropna()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[id_col]).sort_values(id_col).reset_index(drop=True)
    return {int(i): str(n) for i,n in zip(df[id_col], df[nm_col])}

index_to_name = load_label_map_from_csv(labels_csv)

# Model
@st.cache_resource
def load_vit(ckpt, n_classes=25):
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
    if os.path.isfile(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model

model = load_vit(ckpt_path, n_classes=(len(index_to_name) if index_to_name else 25))

tfm = transforms.Compose([
    transforms.Resize((imgsz, imgsz)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top1 = int(np.argmax(probs))
        conf = float(probs[top1])
    name = index_to_name.get(top1, str(top1)) if index_to_name else str(top1)
    st.success(f"Top-1: {name} ({conf*100:.2f}%)")
