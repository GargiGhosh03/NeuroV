
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


## 🖥 Interface Walkthrough

### 🧪 Upload Interface

<table>
  <tr>
    <td style="width: 50%; vertical-align: top; padding: 10px;">
      <p>
        Users can upload <code>.nii</code> or <code>.nii.gz</code> MRI scan files from the T1ce modality.<br><br>
        The model processes each 2D slice, classifies the presence of lesions, and outputs an aggregated result.
      </p>
    </td>
    <td style="width: 50%; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/0efb2e99-0318-427c-b68a-ec819e7fae69" alt="Upload Interface" style="max-width: 100%; border-radius: 10px;" />
    </td>
  </tr>
</table>

---

### 📊 Overview/ Treatments

<table>
  <tr>
    <td style="width: 50%; vertical-align: top; padding: 10px;">
      <p>
        Brain tumor depends upon several factors including the type, size, location of the tumor, and the patient's overall health.<br><br>
        Tells about the different kinds of treatments available for the according to the results. 
      </p>
    </td>
    <td style="width: 50%; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/6a332ce4-f12a-4992-bb8a-b3aefb36810a" alt="Slice-wise Predictions" style="max-width: 100%; border-radius: 10px;" />
    </td>
  </tr>
</table>

---

### ✅ Results

<table>
  <tr>
    <td style="width: 50%; vertical-align: top; padding: 10px;">
      <p>
        All predictions are aggregated into a single classification result:<br><br>
        <strong>"Tumor Detected"</strong> or <strong>"No Tumor Found"</strong>.<br><br>
        A confidence score is also displayed.
      </p>
    </td>
    <td style="width: 50%; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/c005fab3-763e-4c1a-b31b-dc88f8d03908" alt="Final Diagnosis" style="max-width: 100%; border-radius: 10px;" />
    </td>
  </tr>
</table>
