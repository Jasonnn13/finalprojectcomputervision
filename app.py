import io
import os
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_large

MODEL_PATH = os.path.join(os.path.dirname(__file__), "CTScan_ConvNeXtLarge.pth")

def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        return b"git-lfs" in head or head.startswith(b"version https://git-lfs.github.com/spec")
    except Exception:
        return False

@st.cache_resource(show_spinner=True)
def load_model(model_path: str) -> Tuple[nn.Module, int]:
    # Build base model
    model = convnext_large(weights=None)
    # Detect common issue: LFS pointer file not downloaded
    if _is_lfs_pointer(model_path):
        raise RuntimeError(
            "Model file appears to be a Git LFS pointer. Run 'git lfs pull' or 'git lfs checkout' to download the actual binary."
        )
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    num_classes: Optional[int] = None
    for k, v in state_dict.items():
        if k.endswith("classifier.2.weight") and v.ndim == 2:
            num_classes = v.shape[0]
            break
        if k.endswith("classifier.1.weight") and v.ndim == 2:
            num_classes = v.shape[0]
            break
        if k.endswith("fc.weight") and v.ndim == 2:
            num_classes = v.shape[0]
            break

    if num_classes is None:
        linear_keys = [k for k in state_dict.keys() if k.endswith("weight") and state_dict[k].ndim == 2]
        if linear_keys:
            num_classes = state_dict[linear_keys[-1]].shape[0]
        else:
            num_classes = 2 

    in_features = model.classifier[-1].in_features
    new_head = nn.Sequential(
        nn.Flatten(),
        nn.LayerNorm(in_features, eps=1e-6),
        nn.Linear(in_features, num_classes),
    )
    model.classifier = new_head
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, num_classes

CLASS_TO_LABEL = {
    0: "Adenocarcinoma",
    1: "Large Cell Carcinoma",
    2: "Normal",
    3: "Squamous Cell Carcinoma",
}

def default_labels(num_classes: int) -> List[str]:
    mapped = [CLASS_TO_LABEL.get(i, f"Class {i}") for i in range(num_classes)]
    return mapped

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict(model: nn.Module, image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    x = PREPROCESS(image).unsqueeze(0)  # shape (1, C, H, W)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

st.set_page_config(
    page_title="Chest CT Scan Classifier",
    page_icon="🫁",
    layout="centered",
)

# Hero section
st.markdown(
    """
    <div style='text-align:center;'>
      <h1 style='margin-bottom:0'>Chest CT Scan Classifier</h1>
      <p style='color:#6b7280;'>Upload a chest CT image to get model predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model")
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found: CTScan_ConvNeXtLarge.pth")
    else:
        try:
            model, num_classes = load_model(MODEL_PATH)
            st.success(f"Loaded ConvNeXt-Large with {num_classes} classes")
            st.caption(f"Checkpoint: {os.path.basename(MODEL_PATH)}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.caption("If this is an LFS pointer, run 'git lfs pull' in the repo.")

uploaded = st.file_uploader("Upload CT image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is not None and os.path.exists(MODEL_PATH):
    try:
        img_bytes = uploaded.read()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        st.error(f"Failed to read image: {e}")
        image = None

    if image is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Preview")
            st.image(image, caption="Uploaded CT image", use_column_width=True)
        with col2:
            st.subheader("Prediction")
            model, num_classes = load_model(MODEL_PATH)  # ensure loaded
            labels = default_labels(num_classes)
            probs = predict(model, image)
            top_idx = int(np.argmax(probs))
            st.write(f"Prediction: {labels[top_idx]} — {probs[top_idx]*100:.2f}%")

            st.info("Predictions are probabilistic and not a medical diagnosis.")
else:
    st.info("Upload a chest CT image to begin.")


