import torch
import nibabel as nib
import numpy as np
from PIL import Image
from torchvision import transforms

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def normalize_slice(slice):
    slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice) + 1e-5)
    return (slice * 255).astype(np.uint8)

def predict_nii_file(path, model):
    img = nib.load(path).get_fdata()
    predictions = []
    true_labels = []

    for i in range(img.shape[2]):
        slice_img = normalize_slice(img[:, :, i])
        pil_img = Image.fromarray(slice_img).convert("L")
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            prob = output.item()
            predictions.append(prob)

            # Simulated label (since we don't have ground truth for .nii test files)
            true_labels.append(1)  # If you know the file is healthy

    avg_score = np.mean(predictions)
    prediction = "Tumor" if avg_score > 0.5 else "No Tumor"
    confidence = round(avg_score if avg_score > 0.5 else 1 - avg_score, 3)

    # Calculate binary predictions
    predicted_labels = [1 if p > 0.5 else 0 for p in predictions]
    correct = sum([1 for pred, true in zip(predicted_labels, true_labels) if pred == true])
    accuracy = round(correct / len(true_labels), 3)

    return prediction, confidence, accuracy, predictions
