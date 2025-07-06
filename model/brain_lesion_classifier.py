
import os
import nibabel as nib
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random
import torch.nn as nn

# === Path Configuration ===
root_dir = r"E:\CV REQUIREMENTS\BrainlyAI\Brain\MICCAI_BraTS2020_TrainingData"

# === Normalize function ===
def normalize_slice(slice):
    slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice) + 1e-5)
    slice = (slice * 255).astype(np.uint8)
    return slice

# === Load patient slices ===
def load_patient_slices(patient_path, modality="t1ce"):
    slices = []
    patient_id = os.path.basename(patient_path)
    base_name = patient_id

    modality_file = os.path.join(patient_path, f"{base_name}_{modality}.nii")
    seg_file = os.path.join(patient_path, f"{base_name}_seg.nii")

    if not os.path.exists(modality_file) or not os.path.exists(seg_file):
        return []

    img = nib.load(modality_file).get_fdata()
    seg = nib.load(seg_file).get_fdata()

    for i in range(img.shape[2]):
        img_slice = normalize_slice(img[:, :, i])
        mask_slice = seg[:, :, i]
        label = 1 if np.any(mask_slice > 0) else 0
        pil_img = Image.fromarray(img_slice).convert("L")
        slices.append((pil_img, label))

    return slices

# === Load all patients ===
def get_all_slices(root_dir):
    all_data = []
    patients = os.listdir(root_dir)
    for patient in tqdm(patients, desc="Loading patients"):
        path = os.path.join(root_dir, patient)
        slices = load_patient_slices(path)
        all_data.extend(slices)
    print(f"Total slices loaded: {len(all_data)}")
    return all_data

# === Dataset Class ===
class MRIDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

# === Improved Model ===
class BrainLesionClassifier(nn.Module):
    def __init__(self):
        super(BrainLesionClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))

# === Train Script ===
def train():
    all_data = get_all_slices(root_dir)
    random.shuffle(all_data)
    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    val_data = all_data[split:]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = MRIDataset(train_data, transform=transform)
    val_dataset = MRIDataset(val_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = BrainLesionClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation ROC AUC
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs)
                preds.extend(outputs.squeeze().tolist())
                targets.extend(labels.squeeze().tolist())
        roc = roc_auc_score(targets, preds)
        print(f"Validation ROC AUC: {roc:.4f}")

    torch.save(model.state_dict(), "brain_lesion_classifier.pth")

if __name__ == "__main__":
    train()
