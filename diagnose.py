
import cv2
import numpy as np
import os

filepath = 'scans/IMAG0001.JPG'
img = cv2.imread(filepath)
print(f"Original shape: {img.shape}")
print(f"Original mean: {np.mean(img)}")

def check_stats(gamma_val, name):
    print(f"\n--- Checking {name} (Gamma={gamma_val}) ---")
    look_up_table = np.empty((1, 256), np.uint8)
    # Using p = gamma as power
    p = gamma_val
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, p) * 255.0, 0, 255)
    
    img_corrected = cv2.LUT(img, look_up_table)
    mean_val = np.mean(img_corrected)
    print(f"Mean brightness: {mean_val}")
    
    hsv = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Otsu threshold on V
    ret_otsu_v, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu V threshold: {ret_otsu_v}")
    
    # Otsu threshold on S
    ret_otsu_s, _ = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu S threshold: {ret_otsu_s}")
    
    # Try a hard V threshold for High Contrast (assuming flowers are bright)
    # If background mean is 9.7 (std probably ~10-20). 50 is safe.
    # User said "High Contrast".
    mask_high_v = (v > 60).astype(np.uint8) * 255
    mask_high_s = (s > 40).astype(np.uint8) * 255
    final_mask = cv2.bitwise_and(mask_high_v, mask_high_s)
    
    print(f"Hard Threshold (V>60 & S>40) non-zero: {cv2.countNonZero(final_mask)}")
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
        print(f"Top 5 Contours: {areas[:5]}")
        valid_areas = [a for a in areas if a > 50 and a < 1000000] # Exclude massive things
        print(f"Sum Area (50 < a < 1M): {sum(valid_areas)}")
        print(f"Count (50 < a < 1M): {len(valid_areas)}")
    else:
        print("No contours")

check_stats(2.0, "Power 2.0 (Darken)")
check_stats(0.5, "Power 0.5 (Brighten)")
