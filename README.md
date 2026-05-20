# Brain Tumor Classification using Vision Transformer (ViT)

This project implements a state-of-the-art Brain Tumor Classification pipeline using a Vision Transformer (ViT) architecture. It classifies MRI scans into four categories: Glioma, Meningioma, Pituitary, and No Tumor.

## 🚀 Key Features
- **ViT Architecture:** Uses `vit_tiny_patch16_224` for efficient and accurate classification.
- **Enhanced Training:** Includes Early Stopping, LR Scheduling, and Reproducibility.
- **Comprehensive Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- **Interpretability:** Grad-CAM and Feature Map visualizations for transparency.
- **Interactive Dashboard:** Streamlit app for easy inference and visualization.

## 📂 Project Structure
- `train.py`: Main training script with optimizations and history tracking.
- `evaluate.py`: Evaluation script for generating detailed metrics and plots.
- `interpret.py`: Script for feature learning analysis and Grad-CAM generation.
- `app.py`: Streamlit dashboard for user interaction.
- `archive/`: Dataset directory containing Training and Testing folders.
- `requirements.txt`: Project dependencies.

## 🛠️ Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   ```bash
   python train.py
   ```
   *This will save the best model as `best_vit_model.pth` and generate `training_plots.png`.*

3. **Evaluate the Model:**
   ```bash
   python evaluate.py
   ```
   *This will generate `test_metrics.txt`, `confusion_matrix.png`, `roc_curve.png`, etc.*

4. **Run Interpretability Analysis:**
   ```bash
   python interpret.py
   ```
   *Visualizes what the model learns at different depths.*

5. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📊 Feature Learning Analysis
The ViT model learns features in stages:
- **Early Layers:** Focus on edges, lines, and simple MRI textures.
- **Middle Layers:** Identify tumor regions, shapes, and spatial structures.
- **Deep Layers:** Capture high-level pathological features using attention mechanisms.

## 📈 Results
Evaluation results and visualizations are saved in the project root after running `evaluate.py`.
