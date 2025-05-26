import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ResNetFeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove final layer
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image).squeeze()
        return features.numpy()
