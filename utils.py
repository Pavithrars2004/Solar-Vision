import cv2
import numpy as np

def preprocess_image(image):
    """Resize and normalize the input image."""
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return image

def calculate_rooftop_area(mask, pixel_area=1):
    """Calculate rooftop area based on predicted mask."""
    rooftop_pixels = np.sum(mask > 0.5)
    rooftop_area = rooftop_pixels * pixel_area  # Assuming 1 m² per pixel
    return rooftop_area

def estimate_solar_potential(area_m2, irradiance=5, efficiency=0.17):
    """
    Estimate annual solar potential:
    Energy (kWh/year) = Area * Irradiance * Efficiency * 365
    """
    annual_energy_kwh = area_m2 * irradiance * efficiency * 365
    return annual_energy_kwh
