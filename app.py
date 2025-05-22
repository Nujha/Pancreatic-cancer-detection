import streamlit as st
import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import shap
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import warnings
from rbm_utils import DeterministicRBM


from sklearn.neural_network import BernoulliRBM




warnings.filterwarnings("ignore")

st.set_page_config(page_title="Pancreatic CT Classifier", layout="centered")

# Show header and info
st.title("🧠 AI enhanced medical image analysis for early pancreatic cancer detetion")
st.markdown("Upload a **CT scan image** to analyze and explain the classification.")

# Single instance of file uploader
uploaded_file = st.file_uploader("Upload CT scan image (PNG/JPG)", type=["png", "jpg", "jpeg"])

# ========== DCNN Feature Extractor ==========
class DCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(DCNNFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x)

# ========== Grad-CAM ==========
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.features._modules.items():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.cpu().numpy()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# ========== Model and Transforms ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dcnn_model = DCNNFeatureExtractor().to(device)
dcnn_model.eval()

model_data = joblib.load('dbn_model.pkl')
rbm = model_data['rbm']
logistic = model_data['logistic']
scaler = model_data['scaler']


transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor()
])

# ========== Processing After File Upload ==========
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded CT Scan Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    # Grad-CAM
    grad_cam = GradCAM(dcnn_model, target_layer='6')
    cam_mask = grad_cam(img_tensor)
    grad_cam.remove_hooks()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    img_np = np.array(image.convert("RGB"))
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    st.image(overlay, caption="Grad-CAM Heatmap Overlay", use_container_width=True)
    

    # Feature extraction
    with torch.no_grad():
        features = dcnn_model(img_tensor).view(1, -1).cpu().numpy()

    features_scaled = scaler.transform(features)

    # Prediction
    features_transformed = rbm.transform(features_scaled)
    prediction = logistic.predict(features_transformed)[0]
    proba = logistic.predict_proba(features_transformed)[0]


    label_map = {0: "Normal", 1: "Cancer"}
    st.write(f"**Prediction:** {label_map[prediction]}")
    st.write(f"**Prediction Probability:** Normal: {proba[0]:.3f}, Cancer: {proba[1]:.3f}")

    # SHAP
    st.write("### SHAP Explanation for Logistic Regression")

    background = np.tile(features_scaled, (50, 1))  # repeat same input for stable SHAP
    explainer = shap.LinearExplainer(logistic, background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(features_transformed)


    fig, ax = plt.subplots()
    shap.summary_plot(
    shap_values,
    features_transformed,
    feature_names=[f"feat_{i}" for i in range(features_transformed.shape[1])],
    plot_type="bar",
    show=False
)

    st.pyplot(fig)

