import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
import io
import os
import time

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap');

  html, body, [class*="css"] { font-family: 'Satoshi', sans-serif; }

  section[data-testid="stSidebar"] {
    background: #1c1b19;
    border-right: 1px solid #262523;
  }
  section[data-testid="stSidebar"] * { color: #cdccca !important; }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 { color: #f9f8f5 !important; }

  .main { background: #171614; color: #cdccca; }
  h1, h2, h3 { color: #f9f8f5 !important; font-weight: 700; }

  .stButton > button {
    background: #01696f; color: #f9f8f5;
    border: none; border-radius: 6px;
    padding: 0.5rem 1.25rem; font-weight: 500;
    transition: background 180ms ease;
  }
  .stButton > button:hover { background: #0c4e54; }

  .metric-card {
    background: #1c1b19; border: 1px solid #262523;
    border-radius: 8px; padding: 1rem 1.25rem;
    text-align: center; margin-bottom: 0.5rem;
  }
  .metric-val { font-size: 1.75rem; font-weight: 700; color: #4f98a3; }
  .metric-lbl { font-size: 0.8rem; color: #797876; text-transform: uppercase; letter-spacing: 0.05em; }

  .img-header {
    background: #1c1b19; border: 1px solid #262523;
    border-radius: 8px 8px 0 0; padding: 0.5rem 0.75rem;
    font-size: 0.75rem; font-weight: 500; color: #797876;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .img-wrapper {
    border: 1px solid #262523; border-top: none;
    border-radius: 0 0 8px 8px; overflow: hidden;
  }

  .stSelectbox label, .stSlider label, .stFileUploader label {
    color: #cdccca !important; font-weight: 500;
  }

  .stAlert { border-radius: 8px; }

  div[data-testid="stImage"] img { border-radius: 0 0 6px 6px; }

  .sidebar-section {
    background: #22211f; border-radius: 8px;
    padding: 0.75rem 1rem; margin-bottom: 1rem;
    border: 1px solid #2d2c2a;
  }
</style>
""", unsafe_allow_html=True)

# ── Model Registry ─────────────────────────────────────────────────────────────
# Adjust these paths to match your Google Drive structure
MODEL_PATHS = {
    "YOLOv8-n  |  auto (AdamW)": "./yolo_models/nano_auto.pt",
    "YOLOv8-n  |  SGD":          "./yolo_models/nano_sdg.pt",
    "YOLOv8-n  |  Adam":         "./yolo_models/nano_adam.pt",
    "YOLOv8-s  |  AdamW":        "./yolo_models/small_adamw.pt",
}

# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    return YOLO(path)

# ── Drawing helper ─────────────────────────────────────────────────────────────
COLORS = [
    "#4f98a3", "#6daa45", "#e8af34", "#dd6974",
    "#a86fdf", "#fdab43", "#5591c7", "#d163a7",
]

def draw_predictions(image: Image.Image, results, conf_threshold: float):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    boxes = results[0].boxes
    names = results[0].names

    detected = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        label = names[cls_id]
        color = COLORS[cls_id % len(COLORS)]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = img.size

        # Box
        lw = max(2, int(min(w, h) * 0.003))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        # Label background
        text = f"{label} {conf:.0%}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((x1, y1), text, font=font)
        pad = 3
        draw.rectangle(
            [bbox[0]-pad, bbox[1]-pad-lw, bbox[2]+pad, bbox[3]+pad],
            fill=color
        )
        draw.text((x1, y1 - lw - pad), text, fill="white", font=font)
        detected.append({"class": label, "confidence": conf})

    return img, detected

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 YOLO Inference")
    st.markdown("---")

    st.markdown("### Model")
    selected_model_name = st.selectbox(
        "Select trained model",
        options=list(MODEL_PATHS.keys()),
        label_visibility="collapsed",
    )

    st.markdown("### Confidence Threshold")
    conf_threshold = st.slider(
        "Minimum confidence to display",
        min_value=0.05, max_value=0.95,
        value=0.25, step=0.05,
        format="%.2f",
        label_visibility="collapsed",
    )
    st.caption(f"Showing detections ≥ **{conf_threshold:.0%}** confidence")

    st.markdown("---")
    model_path = MODEL_PATHS[selected_model_name]
    if os.path.exists(model_path):
        st.success("✅ Model weights found")
    else:
        st.error("❌ Model path not found\nCheck MODEL_PATHS in the script")

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("# Object Detection Inference")
st.markdown(
    f"Model: **{selected_model_name}** &nbsp;·&nbsp; "
    f"Confidence threshold: **{conf_threshold:.0%}**",
)
st.markdown("---")

uploaded_files = st.file_uploader(
    "Upload images (PNG, JPG, JPEG — multiple allowed)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("⬆️  Upload one or more images using the uploader above to start inference.")
    st.stop()

# Load model
with st.spinner(f"Loading {selected_model_name} …"):
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ── Per-image results ──────────────────────────────────────────────────────────
total_detections = 0
total_time_ms = 0

for idx, file in enumerate(uploaded_files):
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    t0 = time.perf_counter()
    results = model(image, verbose=False)
    elapsed = (time.perf_counter() - t0) * 1000
    total_time_ms += elapsed

    annotated_img, detected = draw_predictions(image, results, conf_threshold)
    total_detections += len(detected)

    st.markdown(f"### Image {idx + 1} — `{file.name}`")

    # Metrics row
    col_m1, col_m2, col_m3, _ = st.columns([1, 1, 1, 3])
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{len(detected)}</div>
          <div class="metric-lbl">Detections</div>
        </div>""", unsafe_allow_html=True)
    with col_m2:
        avg_conf = np.mean([d["confidence"] for d in detected]) if detected else 0
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{avg_conf:.0%}</div>
          <div class="metric-lbl">Avg Conf</div>
        </div>""", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{elapsed:.0f}ms</div>
          <div class="metric-lbl">Inference</div>
        </div>""", unsafe_allow_html=True)

    # Images side-by-side
    col_orig, col_pred = st.columns(2)
    with col_orig:
        st.markdown('<div class="img-header">📷 Original</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col_pred:
        n = len(detected)
        label_str = f"{n} detection{'s' if n != 1 else ''}"
        st.markdown(f'<div class="img-header">🎯 Predictions — {label_str}</div>', unsafe_allow_html=True)
        st.image(annotated_img, use_container_width=True)

    # Detection list
    if detected:
        from collections import Counter
        counts = Counter(d["class"] for d in detected)
        tags = "  ".join(
            f"`{cls}` × {cnt}" for cls, cnt in sorted(counts.items())
        )
        st.markdown(f"**Detected classes:** {tags}")
    else:
        st.warning(f"No detections above {conf_threshold:.0%} confidence threshold.")

    st.markdown("---")

# ── Session summary ────────────────────────────────────────────────────────────
if len(uploaded_files) > 1:
    st.markdown("## Session Summary")
    s1, s2, s3, s4 = st.columns(4)
    for col, val, lbl in [
        (s1, len(uploaded_files), "Images Processed"),
        (s2, total_detections,   "Total Detections"),
        (s3, f"{total_time_ms/len(uploaded_files):.0f}ms", "Avg Inference"),
        (s4, f"{total_detections/len(uploaded_files):.1f}", "Detections / Image"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
