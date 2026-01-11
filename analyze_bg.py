import cv2
import numpy as np

def analyze_background(filepath):
    img = cv2.imread(filepath)
    if img is None:
        print("Could not read image")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    
    # Sample borders
    margin = 50
    borders = []
    borders.append(hsv[0:margin, :]) # Top
    borders.append(hsv[h-margin:h, :]) # Bottom
    borders.append(hsv[:, 0:margin]) # Left
    borders.append(hsv[:, w-margin:w]) # Right
    
    border_pixels = np.vstack([b.reshape(-1, 3) for b in borders])
    
    median_hsv = np.median(border_pixels, axis=0)
    mean_hsv = np.mean(border_pixels, axis=0)
    std_hsv = np.std(border_pixels, axis=0)
    
    print(f"File: {filepath}")
    print(f"Median HSV: {median_hsv}")
    print(f"Mean HSV: {mean_hsv}")
    print(f"Std HSV: {std_hsv}")
    
    # Check if V is high (Bright background)
    if median_hsv[2] > 60:
        print("Detected: BRIGHT Background")
    else:
        print("Detected: DARK Background")

analyze_background('pink_background_sample.png')
