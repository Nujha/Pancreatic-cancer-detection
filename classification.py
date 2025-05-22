import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import classification_report
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import random
import os
import joblib
from pathlib import Path
import cv2
from PIL import Image
from torch.utils.data import Dataset

# ========= Reproducibility =========
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ========= Device =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Dataset Class =========
class PancreaticCancerDataset(Dataset):
    def __init__(self, ct_dir, mask_dir, transform=None):
        self.ct_dir = Path(ct_dir)
        self.mask_dir = Path(mask_dir)
        self.ct_images = sorted(self.ct_dir.rglob('*.png'))
        self.masks = sorted(self.mask_dir.rglob('*.png'))
        self.transform = transform

        if len(self.ct_images) != len(self.masks):
            min_len = min(len(self.ct_images), len(self.masks))
            self.ct_images = self.ct_images[:min_len]
            self.masks = self.masks[:min_len]

    def __len__(self):
        return len(self.ct_images)

    def __getitem__(self, idx):
        ct_image = cv2.imread(str(self.ct_images[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        ct_image = ct_image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        if self.transform:
            ct_image = Image.fromarray(ct_image)
            ct_image = self.transform(ct_image)
        label = 1 if np.sum(mask) > 0 else 0
        return ct_image, torch.tensor(label, dtype=torch.long)

# ========= DCNN Model =========
class DCNNClassifier(nn.Module):
    def __init__(self):
        super(DCNNClassifier, self).__init__()
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========= Paths =========
ct_dir = 'C:/preprocessed_pngs'
mask_dir = 'C:/preprocessed_masks'

# ========= Transforms =========
transform_class = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
])

# ========= Load Dataset =========
dataset = PancreaticCancerDataset(ct_dir, mask_dir, transform=transform_class)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# ========= Train DCNN =========
model = DCNNClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# ========= Save Trained DCNN =========
torch.save(model.state_dict(), "trained_dcnn.pth")
print("✅ Trained DCNN model saved as 'trained_dcnn.pth'")

# ========= Feature Extraction =========
def extract_features(dataloader, model):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model.features(x).view(x.size(0), -1)
            features.append(out.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(labels)

X_train, y_train = extract_features(train_loader, model)
X_val, y_val = extract_features(val_loader, model)

# ========= DBN Training =========
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_scaled, y_train)

rbm = BernoulliRBM(n_components=64, learning_rate=0.01, n_iter=30, random_state=42)
logistic = LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=42)

X_train_transformed = rbm.fit_transform(X_train_bal)
logistic.fit(X_train_transformed, y_train_bal)

# ========= Evaluation =========
X_val_transformed = rbm.transform(X_val_scaled)
y_pred = logistic.predict(X_val_transformed)
print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred, target_names=['Normal', 'Cancer']))

# ========= Save DBN Pipeline =========
joblib.dump({'rbm': rbm, 'logistic': logistic, 'scaler': scaler}, 'dbn_model.pkl')
print("✅ DBN model and scaler saved as 'dbn_model.pkl'")
