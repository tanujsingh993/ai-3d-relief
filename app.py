import streamlit as st
import torch
import cv2
import numpy as np
from stl import mesh
import io
from streamlit_stl import stl_from_file

# --- Setup ---
st.set_page_config(page_title="AI 3D Relief", layout="wide")

@st.cache_resource
def load_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transform

model, transform = load_model()

st.title("🏗️ Real-Time AI Relief Maker")

# --- Sidebar Controls ---
st.sidebar.header("3D Adjustments")
contrast = st.sidebar.slider("Depth Intensity", 0.1, 5.0, 1.5)
blur = st.sidebar.slider("Smoothing", 1, 15, 5, step=2)
base_thick = st.sidebar.slider("Base Thickness (mm)", 0.5, 10.0, 2.0)
res_downscale = st.sidebar.select_slider("Preview Resolution", options=[8, 4, 2], value=4, help="Higher = Faster, Lower = More Detail")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and process
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Original Photo", use_container_width=True)

    # AI Depth Logic
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    
    # Refine
    processed = cv2.GaussianBlur(depth, (blur, blur), 0)
    processed = np.clip(processed * contrast, 0, 255)
    
    # Generate STL for Viewer
    d = res_downscale
    data = cv2.resize(processed, (processed.shape[1]//d, processed.shape[0]//d))
    rows, cols = data.shape
    
    vertices = []
    for y in range(rows):
        for x in range(cols):
            vertices.append([x, y, base_thick + (data[y, x] * 0.04)])
    
    vertices = np.array(vertices)
    faces = []
    for y in range(rows-1):
        for x in range(cols-1):
            v0, v1 = y*cols+x, y*cols+x+1
            v2, v3 = (y+1)*cols+x, (y+1)*cols+x+1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    relief_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3): relief_mesh.vectors[i][j] = vertices[f[j], :]
    
    # Save to temp file for the viewer
    relief_mesh.save("preview.stl")

    with col2:
        st.subheader("Interactive 3D Preview")
        # The Viewer Component
        stl_from_file("preview.stl", color="#00d4ff", material="material", auto_rotate=False)
        
        with open("preview.stl", "rb") as f:
            st.download_button("📥 Download This STL", f, file_name="ai_relief.stl")
