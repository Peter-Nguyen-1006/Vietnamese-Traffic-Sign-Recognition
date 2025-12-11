import os, time
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Traffic Sign - YOLOv8 Demo", layout="wide")
st.title("🚦 Vietnamese Traffic Sign – YOLOv8-CLS Demo")

weights = st.sidebar.text_input("YOLOv8 weights (.pt)", value="runs/classify/exp/weights/best.pt")
imgsz = st.sidebar.number_input("Input size", min_value=64, max_value=640, value=224, step=32)

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(weights)

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded and os.path.isfile(weights):
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)
    r = model.predict(img, imgsz=imgsz, verbose=False)[0]
    top1 = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    name = r.names[top1]
    st.success(f"Top-1: {name} ({conf*100:.2f}%)")
