import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from utils.preprocessing import predict_nii_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# === Define the model class ===
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

# === Load the model ===
model = BrainLesionClassifier()
model.load_state_dict(torch.load(r'E:\CV REQUIREMENTS\BrainlyAI\Brain\brain_lesion_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    accuracy = None
    slice_scores = []

    if request.method == 'POST':
        file = request.files.get('nii_file')
        if not file or file.filename == '':
            return "No file uploaded", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction, confidence, accuracy, slice_scores = predict_nii_file(filepath, model)

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        accuracy=accuracy,
        slice_scores=slice_scores
    )
    
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/treatments")
def treatments():
    return render_template("treatments.html")

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

@app.route("/consult")
def consult():
    return render_template("consult.html")


if __name__ == '__main__':
    app.run(debug=True)
