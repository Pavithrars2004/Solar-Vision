# ☀️ Solar Vision: Rooftop Solar Potential Estimator

An interactive, end-to-end web application that estimates rooftop solar potential from satellite images using computer vision.

This project combines **image segmentation, rooftop area estimation, and solar energy potential calculation** into a user-friendly web app built with **Streamlit**.

---

## 🚀 Project Features

- ✅ Upload satellite images of buildings.
- ✅ Automatically detect rooftops using a CNN-based segmentation model.
- ✅ Estimate rooftop area (m²).
- ✅ Calculate potential annual solar energy generation (kWh/year).
- ✅ Clean, interactive Streamlit interface.
- ✅ Modular code for easy improvements and future API integration.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Deep Learning:** PyTorch, CNN (U-Net also can be used but i used CNN)
- **Libraries:** OpenCV, NumPy, Matplotlib, Pillow

---

## 📂 Project Structure

```text
solar_vision/
├── app.py               # Streamlit web app
├── model.py             # Rooftop detection model (CNN architecture)
├── utils.py             # Image processing and solar calculation utilities
├── requirements.txt     # Project dependencies
├── sample_images/       # Sample satellite images
│   ├── sample1.jpg
│   └── sample2.jpg
└── README.md            # Project documentation
