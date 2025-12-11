import io
from typing import List, Optional, Tuple

import numpy as np
import requests
import streamlit as st
import torch
torch.classes.__path__ = []
import torch.nn as nn
from PIL import Image

from pathlib import Path

hugging_face_model_link = "https://huggingface.co/spaces/Jasonnn13/CTScan_ConvNeXtLarge.pth/resolve/main/CTScan_ConvNeXtLarge.pth"
MODEL_FILENAME = "CTScan_ConvNeXtLarge.pth"

# Fixed class mapping provided by user
CLASS_TO_LABEL = {
	0: "Adenocarcinoma",
	1: "Large Cell Carcinoma",
	2: "Normal",
	3: "Squamous Cell Carcinoma",
}


def _infer_num_classes_from_state(state_dict: dict) -> Optional[int]:
	candidates = [
		"classifier.2.weight",
		"head.fc.weight",
		"fc.weight",
		"classifier.weight",
	]
	for k in candidates:
		if k in state_dict:
			return int(state_dict[k].shape[0])
	# Try to find any linear layer weight at the tail of classifier
	keys = [k for k in state_dict.keys() if k.endswith(".weight")]
	for k in keys:
		if ".classifier" in k or ".head" in k or k.endswith("fc.weight"):
			try:
				return int(state_dict[k].shape[0])
			except Exception:
				pass
	return None


def _infer_class_names(ckpt: dict, num_classes: int) -> List[str]:
	# Common patterns
	for key in ("classes", "class_names", "labels"):
		if isinstance(ckpt.get(key), (list, tuple)):
			return list(ckpt[key])
	if isinstance(ckpt.get("idx_to_class"), dict):
		# Ensure ordered by index
		mapping = ckpt["idx_to_class"]
		try:
			return [mapping[i] for i in range(len(mapping))]
		except Exception:
			# Fallback arbitrary order
			return list(mapping.values())
	if isinstance(ckpt.get("class_to_idx"), dict):
		inv = sorted(ckpt["class_to_idx"].items(), key=lambda x: x[1])
		return [name for name, _ in inv]
	return [f"Class {i}" for i in range(num_classes)]


@st.cache_resource(show_spinner=True)
def load_model(weights_path: str) -> Tuple[nn.Module, List[str]]:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = torch.load(weights_path, map_location=device)
	if isinstance(ckpt, dict):
		state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
	else:
		state_dict = ckpt

	# Prefer fixed mapping if provided, otherwise infer
	if CLASS_TO_LABEL:
		num_classes = len(CLASS_TO_LABEL)
	else:
		num_classes = _infer_num_classes_from_state(state_dict) or 2

	model = None
	errors = []

	# Try torchvision ConvNeXt Large first
	try:
		from torchvision.models import convnext_large
		tv_model = convnext_large(weights=None)
		in_features = tv_model.classifier[2].in_features
		tv_model.classifier[2] = nn.Linear(in_features, num_classes)
		tv_model.load_state_dict(state_dict, strict=False)
		model = tv_model
	except Exception as e:
		errors.append(f"torchvision load failed: {e}")

	if model is None:
		raise RuntimeError(
			"Failed to load model with the provided weights. "
			+ " ; ".join(errors)
		)

	model.to(device)
	model.eval()

	if CLASS_TO_LABEL and len(CLASS_TO_LABEL) == num_classes:
		class_names = [CLASS_TO_LABEL[i] for i in range(num_classes)]
	else:
		class_names = _infer_class_names(ckpt if isinstance(ckpt, dict) else {}, num_classes)
	return model, class_names


def preprocess_image(img: Image.Image) -> torch.Tensor:
	# Ensure RGB
	if img.mode != "RGB":
		img = img.convert("RGB")
	# Resize to 224 while keeping aspect ratio via center-crop like behavior
	img = img.resize((224, 224))

	arr = np.array(img).astype("float32") / 255.0
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	arr = (arr - mean) / std
	arr = np.transpose(arr, (2, 0, 1))
	tensor = torch.from_numpy(arr)
	return tensor


def predict(model: nn.Module, tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
	device = next(model.parameters()).device
	with torch.no_grad():
		logits = model(tensor.unsqueeze(0).to(device))
		if isinstance(logits, (list, tuple)):
			logits = logits[0]
		probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
		idx = int(np.argmax(probs))
		conf = float(probs[idx])
	return idx, conf, probs


st.set_page_config(page_title="CT Scan Classifier", page_icon="🩺", layout="centered")

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    /* Main Background & Fonts */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        font-weight: 700;
        color: #0f52ba; /* Medical Blue */
    }
    
    /* Info Cards Styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        /* Generic adjustment for nested blocks if needed */
    }
    
    /* Custom Button for "Start Detecting" (Anchor Link) */
    a.custom-btn {
        display: inline-block;
        padding: 0.6em 1.2em;
        margin-top: 20px;
        color: #ffffff !important;
        background-color: #ff4b4b;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        text-align: center;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    a.custom-btn:hover {
        background-color: #ff3333;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    /* Style for the metrics/prediction result */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Resolve static asset directory robustly (works locally and on Streamlit Cloud)
APP_DIR = Path(__file__).parent.resolve()
_public_candidates = [
	APP_DIR / "public",
	Path.cwd() / "public",
	APP_DIR.parent / "public",
]
PUBLIC_DIR = next((p for p in _public_candidates if p.exists()), _public_candidates[0])
MODEL_PATH = APP_DIR / MODEL_FILENAME


def _download_weights(url: str, destination: Path) -> None:
	# Stream download to avoid holding large weights in memory
	response = requests.get(url, stream=True, timeout=60)
	response.raise_for_status()
	destination.parent.mkdir(parents=True, exist_ok=True)
	with destination.open("wb") as fh:
		for chunk in response.iter_content(chunk_size=8192):
			if chunk:
				fh.write(chunk)


def _ensure_model_file() -> Path:
	if MODEL_PATH.exists():
		return MODEL_PATH
	_download_weights(hugging_face_model_link, MODEL_PATH)
	return MODEL_PATH

# --- HERO SECTION ---
st.title("Detect Chest Cancer with CTSense")
st.caption("Fast, Accurate, and Effortless!")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
		"""
		<div style="font-size: 1.1em; color: #444; line-height: 1.6;">
		Welcome to the future of chest cancer detection. With the power of <b>CTSense</b>, 
		you can analyze your <b>CT scans</b> with just one click and receive fast, reliable insights 
		powered by advanced AI technology.
		<br><br>
		Start your scan now and experience precision made simple.
		</div>
		""", unsafe_allow_html=True
	)
    # Replaced st.button with an HTML anchor link styled as a button
    st.markdown('<a href="#prediction-section" class="custom-btn">Start Detecting</a>', unsafe_allow_html=True)

with col2:
	# Prefer local static image if present; fallback to remote URL
	hero_local = PUBLIC_DIR / "1.png"
	st.image(str(hero_local), width='stretch')

# --- INFO SECTION ---
st.divider()
st.header("What You Need to Know About Chest Cancer")

st.subheader("What Is Chest Cancer?")
st.write(
	"Chest cancer refers to several types of cancers that form in the tissues of the lungs. "
	"These cancers grow uncontrollably and can interfere with your breathing, oxygen levels, and overall health. "
	"Some types grow slowly, while others spread quickly. Early detection is crucial."
)

st.subheader("Main Types of Chest Cancer")
st.caption("In our system, we detect these categories:")

# Row 1: Adenocarcinoma | Large Cell Carcinoma
row1_left, row1_right = st.columns(2)
with row1_left:
	with st.container(border=True):
		st.subheader("Adenocarcinoma")
		st.write(
			"A common type of lung cancer that starts in the glandular cells. "
			"It often grows in the outer parts of the lungs and is more likely to appear in non-smokers than other types."
		)
with row1_right:
	with st.container(border=True):
		st.subheader("Large Cell Carcinoma")
		st.write(
			"A more aggressive and large cancer that can appear anywhere in the lungs. "
			"It grows and spreads faster and is usually harder to treat if found late."
		)

# Row 2: Squamous Cell Carcinoma | Normal
row2_left, row2_right = st.columns(2)
with row2_left:
	with st.container(border=True):
		st.subheader("Squamous Cell Carcinoma")
		st.write(
			"This type begins in the thin, flat cells lining the airways. "
			"It often develops in the center of the lungs and is strongly linked to smoking."
		)
with row2_right:
	with st.container(border=True):
		st.subheader("Normal")
		st.write(
			"No signs of detectable cancer were found based on the uploaded scan. "
			"The AI did not identify any suspicious growths (cancer)."
		)

st.subheader("What Happens if It’s Left Untreated?")
st.write(
	"Without treatment, chest cancer can spread to other organs, reduce lung function, "
	"cause severe breathing issues, and become life-threatening. Early diagnosis significantly improves "
	"treatment options and survival rates."
)

st.subheader("How Do You Detect It?")
st.write(
	"Chest cancer often begins with mild or unclear symptoms like coughing, chest pain, or fatigue. "
	"Because these signs can be easily missed, doctors rely on **CT scans** to spot abnormalities."
)
st.write(
	"With CTSense AI, you can upload your chest scan and receive a fast, AI-powered analysis that helps identify "
	"the presence of cancer types such as Adenocarcinoma, Large Cell Carcinoma, and Squamous Cell Carcinoma."
)

st.divider()

# --- PREDICTION / CLASSIFIER SECTION ---
# Add an invisible anchor for the button to scroll to
st.markdown('<div id="prediction-section"></div>', unsafe_allow_html=True)

st.title("CT Scan Classifier (ConvNeXt Large)")

# Sidebar for Model Info & Graphs
with st.sidebar:
	st.subheader("CTSense")
	st.write("Using weights: `CTScan_ConvNeXtLarge.pth`")
 
	st.link_button("GitHub Repository", "https://github.com/Jasonnn13/FinalProjectComputerVision")

	st.subheader("Training Curves")
	shown_any = False
	for rel, label in [    
		("acc.png", "Accuracy"),
		("loss.png", "Loss"),
	]:
		img_path = PUBLIC_DIR / rel
		st.caption(f"{label} (from {img_path.name})")
		st.image(str(img_path), width='stretch')
		shown_any = True

	if not shown_any:
		st.caption("Place images like public/acc.png and public/loss.png to display here.")


@st.cache_resource(show_spinner=False)
def _load_once():
	weights_path = _ensure_model_file()
	return load_model(str(weights_path))


try:
	model, class_names = _load_once()
except Exception as e:
	st.error("Failed to load model. See details below.")
	st.exception(e)
	st.stop()


uploaded = st.file_uploader(
	"Upload CT image (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False
)

if uploaded is not None:
	image_bytes = uploaded.read()
	img = Image.open(io.BytesIO(image_bytes))
	st.image(img, caption="Uploaded Image", width='stretch')

	if st.button("Predict", type="primary"):
		with st.spinner("Running inference..."):
			tensor = preprocess_image(img)
			idx, conf, probs = predict(model, tensor)

		pred_label = class_names[idx] if idx < len(class_names) else f"Class {idx}"

		st.markdown("---")
		st.subheader("Prediction Result")
		col_res1, col_res2 = st.columns(2)
		with col_res1:
			st.success(f"**{pred_label}**")
		with col_res2:
			st.metric("Confidence", f"{conf:.2%}")

else:
	st.info("Please upload an image to begin.")
