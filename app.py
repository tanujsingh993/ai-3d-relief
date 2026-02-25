import streamlit as st
import torch
import cv2
import numpy as np
from stl import mesh
import io

# --- AI Model Setup ---
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return model, transform

model, transform = load_model()

st.title("🌐 AI 3D Relief Web-Maker")
st.write("Upload a photo to generate a 3D STL file for printing.")

# --- Sidebar Settings ---
st.sidebar.header("Settings")
contrast = st.sidebar.slider("Depth Intensity", 0.1, 5.0, 1.5)
blur = st.sidebar.slider("Smoothing", 1, 15, 5, step=2)
base_thickness = st.sidebar.slider("Base Plate (mm)", 0.0, 10.0, 2.0)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate 3D Model"):
        with st.spinner("AI is analyzing depth..."):
            # 1. AI Processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb)
            with torch.no_grad():
                prediction = model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
                ).squeeze()
            
            depth = prediction.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            
            # 2. Refine (Blur & Contrast)
            processed = cv2.GaussianBlur(depth, (blur, blur), 0)
            processed = np.clip(processed * contrast, 0, 255)
            
            # 3. STL Creation (Downscaled for web speed)
            d = 4 # Downscale more for web to save bandwidth
            data = cv2.resize(processed, (processed.shape[1]//d, processed.shape[0]//d))
            rows, cols = data.shape
            z_scale = 0.04
            
            vertices = []
            for y in range(rows):
                for x in range(cols):
                    vertices.append([x, y, base_thickness + (data[y, x] * z_scale)])
            
            vertices = np.array(vertices)
            faces = []
            for y in range(rows - 1):
                for x in range(cols - 1):
                    v0, v1 = y * cols + x, y * cols + x + 1
                    v2, v3 = (y + 1) * cols + x, (y + 1) * cols + x + 1
                    faces.append([v0, v1, v2])
                    faces.append([v1, v3, v2])

            # Save to Buffer
            stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3): stl_mesh.vectors[i][j] = vertices[f[j], :]
            
            stl_io = io.BytesIO()
            stl_mesh.save("temp.stl") # Library requires a filename usually
            with open("temp.stl", "rb") as f:
                stl_io = f.read()

            st.success("3D Relief Ready!")

            st.download_button(label="📥 Download STL File", data=stl_io, file_name="ai_relief.stl", mime="application/sla")
