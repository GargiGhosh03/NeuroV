
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

