import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from interpret import get_gradcam, load_model

# ✅ Page Config
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
MODEL_PATH = "best_vit_model.pth"

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model()
    return None

def predict(model, img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    return pred.item(), confidence.item(), probs[0].numpy()

st.title("🧠 Brain Tumor Classification")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please train first.")
else:
    model = get_model()
    tab1, tab2 = st.tabs(["Inference", "Metrics"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload MRI...", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, width=300)
            idx, conf, probs = predict(model, image)
            st.write(f"### Result: {class_names[idx]} ({conf*100:.2f}%)")
            
            # Grad-CAM
            input_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ])(image).unsqueeze(0)
            cam = get_gradcam(model, input_tensor, model.blocks[-1].norm1, np.float32(image.resize((224, 224)))/255)
            st.image(cam, caption="Grad-CAM")

    with tab2:
        if os.path.exists("training_plots.png"): st.image("training_plots.png")
        if os.path.exists("confusion_matrix.png"): st.image("confusion_matrix.png")
