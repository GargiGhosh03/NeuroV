
# 🧠 NeuroV – Brain Tumor Detection from MRI Scans

**NeuroV** is a deep learning-based web application that detects the presence of brain tumors from MRI scans in `.nii` or `.nii.gz` format. The system is built with **Flask** for the web backend and **PyTorch** for the inference engine, and provides slice-level confidence visualization using **Chart.js**.

---

## 🌟 Features

- ✅ Upload NIfTI MRI scans (`.nii`, `.nii.gz`)
- ✅ Predict presence of brain tumor using a trained CNN model
- ✅ Display overall prediction confidence and accuracy
- ✅ Visualize per-slice confidence scores in a responsive bar chart
- ✅ Clean, responsive UI with navigation bar and feedback indicators

---

## 🧠 Model

The model used is a custom **Convolutional Neural Network** (CNN) trained on the **BraTS 2020** dataset. It classifies each 2D slice from the T1ce modality and aggregates the predictions.

> 📦 The model file `brain_lesion_classifier.pth` is loaded using PyTorch's `state_dict`.


<br> <div align="center"> <div style="border: 2px solid #ccc; border-radius: 12px; padding: 10px; max-width: 90%; background-color: #f9f9f9;"> <img src="https://github.com/user-attachments/assets/0efb2e99-0318-427c-b68a-ec819e7fae69" alt="Model Screenshot" style="max-width:100%; border-radius: 8px;" /> <p style="text-align: center; font-style: italic; color: #555; margin-top: 10px;"> 🧪 “This is the model inference interface where users can upload MRI slices (.nii or .nii.gz) and receive tumor predictions.” </p> </div> </div>


<br> <div align="center"> <div style="border: 2px solid #ccc; border-radius: 12px; padding: 10px; max-width: 90%; background-color: #f9f9f9;"> <img width="1831" height="843" alt="Screenshot 2025-07-07 130406" src="https://github.com/user-attachments/assets/6a332ce4-f12a-4992-bb8a-b3aefb36810a" /> <p style="text-align: center; font-style: italic; color: #555; margin-top: 10px;">


<br> <div align="center"> <div style="border: 2px solid #ccc; border-radius: 12px; padding: 10px; max-width: 90%; background-color: #f9f9f9;"> <img width="1844" height="850" alt="Screenshot 2025-07-07 130423" src="https://github.com/user-attachments/assets/e7734c80-d080-4a47-95b1-cc98b5ca342e" /> <p style="text-align: center; font-style: italic; color: #555; margin-top: 10px;">
