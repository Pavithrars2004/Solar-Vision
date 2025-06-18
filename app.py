import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import load_model, predict
from utils import preprocess_image, calculate_rooftop_area, estimate_solar_potential

# Streamlit UI
st.set_page_config(page_title="Solar Vision: Rooftop Solar Estimator", layout="centered")
st.title("☀️ Solar Vision: Rooftop Solar Potential Estimator")

# Upload Image
uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_array = np.array(image)
    preprocessed_image = preprocess_image(img_array)

    # Load model and predict
    model = load_model()
    mask = predict(model, preprocessed_image)

    # Calculate rooftop area and solar potential
    rooftop_area = calculate_rooftop_area(mask)
    solar_potential = estimate_solar_potential(rooftop_area)

    st.subheader(f"Estimated Rooftop Area: {rooftop_area:.2f} m²")
    st.subheader(f"Estimated Annual Solar Potential: {solar_potential:.2f} kWh/year")

    st.subheader("Rooftop Detection Mask")
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap='gray')
    ax.set_title('Predicted Rooftop Mask')
    st.pyplot(fig)

    st.info("Note: The current model is untrained and serves as a prototype structure. For real-world use, train with labeled rooftop datasets.")
