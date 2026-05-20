import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ✅ Config
device = torch.device("cpu")
MODEL_PATH = "best_vit_model.pth"

def load_model():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_gradcam(model, input_tensor, target_layer, rgb_img):
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

def visualize_feature_maps(model, input_tensor, layer_idx):
    with torch.no_grad():
        x = model.patch_embed(input_tensor)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        
        for i, block in enumerate(model.blocks):
            x = block(x)
            if i == layer_idx:
                break
        
        features = x[:, 1:, :]
        b, n, c = features.shape
        w = int(np.sqrt(n))
        features = features.reshape(b, w, w, c).permute(0, 3, 1, 2)
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < c:
                ax.imshow(features[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
        plt.suptitle(f'Feature Maps from Block {layer_idx}')
        return fig

def run_interpretability(image_path, output_prefix="interpret"):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    rgb_img = np.float32(img.resize((224, 224))) / 255
    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor_norm = transforms.Normalize([0.5] * 3, [0.5] * 3)(input_tensor)
    
    target_layer = model.blocks[-1].norm1
    cam_result = get_gradcam(model, input_tensor_norm, target_layer, rgb_img)
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_result)
    plt.title("Grad-CAM: High-Level Pathological Features")
    plt.axis('off')
    plt.savefig(f"{output_prefix}_gradcam.png")
    
    fig_early = visualize_feature_maps(model, input_tensor_norm, layer_idx=0)
    fig_early.savefig(f"{output_prefix}_early_features.png")
    
    fig_mid = visualize_feature_maps(model, input_tensor_norm, layer_idx=len(model.blocks)//2)
    fig_mid.savefig(f"{output_prefix}_mid_features.png")
    
    fig_deep = visualize_feature_maps(model, input_tensor_norm, layer_idx=len(model.blocks)-1)
    fig_deep.savefig(f"{output_prefix}_deep_features.png")
    print(f"✅ Interpretability visualizations saved with prefix {output_prefix}")

if __name__ == "__main__":
    sample_img = "archive/Testing/glioma_tumor/image(1).jpg"
    if os.path.exists(sample_img):
        run_interpretability(sample_img)
    else:
        print("Sample image not found for testing interpret.py")
