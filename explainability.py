import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax().item()

        loss = output[:, class_idx]
        loss.backward()

        grads = self.gradients
        acts = self.activations

        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        for i in range(acts.shape[1]):
            acts[:, i, :, :] *= pooled_grads[i]

        heatmap = acts.mean(dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap = heatmap / np.max(heatmap)

        return heatmap


class SHAPExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.Explainer(self.model.predict, shap.maskers.Independent(np.zeros((1, 2048))))

    def explain(self, feature_vector):
        shap_values = self.explainer(feature_vector.reshape(1, -1))
        return shap_values
