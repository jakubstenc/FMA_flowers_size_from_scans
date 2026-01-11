import cv2
import numpy as np
import os
import csv
import glob
import zipfile
import roi_writer

# Constants
DPI = 1200
MM2_PER_PIXEL = (25.4 / DPI) ** 2

def process_image(filepath, output_dir=None, bg_mode='dark'):
    filename = os.path.basename(filepath)
    print(f"Processing {filename} (Mode: {bg_mode})...")
    
    # Load image
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Could not read {filepath}")
        return None, None, None
    
    # Gamma Correction - BRIGHTEN (Gamma < 1.0)
    gamma = 0.5 
    look_up_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    img_corrected = cv2.LUT(img, look_up_table)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    # --- LABEL DETECTION (Shape Based) ---
    # Label is white paper.
    
    mask_label_search = None
    if bg_mode == 'pink':
        # Pink background has High S (~175). Label has Low S (~0-30).
        # Both are High V.
        # So we search for Low S regions.
        mask_label_search = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Enforce High V just to be sure
        mask_v_high = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)[1]
        mask_label_search = cv2.bitwise_and(mask_label_search, mask_v_high)
        
    else:
        # Dark Background (Black). Label is Bright (High V).
        mask_label_search = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)[1]
    
    contours_bright, _ = cv2.findContours(mask_label_search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    label_contour = None

    if contours_bright:
        # Find largest bright object
        largest_cnt = max(contours_bright, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        
        # Heuristic: Label is Big (> 5000) and roughly Rectangular (Solidity > 0.8)
        if area > 5000:
            hull = cv2.convexHull(largest_cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if solidity > 0.8:
                label_contour = largest_cnt


    # --- FLOWER DETECTION ---
    mask_objects = None
    
    if bg_mode == 'pink':
        # Pink Detection Logic
        # Pink background median was approx H=161, S=175, V=216
        # H range: 140-180 (Pink/Red/Magenta)
        # S range: > 50 (Saturated)
        # V range: > 100 (Bright)
        lower_pink = np.array([140, 50, 100])
        upper_pink = np.array([180, 255, 255])
        
        mask_bg = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Also could be some grayish/white parts of background if uneven?
        # For now, assume pink is dominant.
        
        mask_objects = cv2.bitwise_not(mask_bg)
        
        # Clean up: apply Value threshold too? 
        # Background is V~216. Flowers might be darker or lighter.
        # But `mask_objects` now contains "Everything NOT Pink".
        
    else:
        # Dark Background Logic (Default)
        # Flowers are simply "Not Background" (V > Threshold)
        # Threshold V > 60 (Dark enough to exclude background, conservative)
        mask_objects = cv2.threshold(v, 60, 255, cv2.THRESH_BINARY)[1]
    
    # --- BORDER MASKING ---
    # Mask out the edges of the image (e.g., 100 pixels) to avoid scanner artifacts
    h, w = mask_objects.shape
    border_margin = 100
    cv2.rectangle(mask_objects, (0, 0), (w, border_margin), 0, -1) # Top
    cv2.rectangle(mask_objects, (0, h-border_margin), (w, h), 0, -1) # Bottom
    cv2.rectangle(mask_objects, (0, 0), (border_margin, h), 0, -1) # Left
    cv2.rectangle(mask_objects, (w-border_margin, 0), (w, h), 0, -1) # Right
    
    # Remove Label from mask
    if label_contour is not None:
        # Create a mask for the label
        # Use Bounding Rect + Margin to completely cover the jagged paper edges
        x, y, w_rect, h_rect = cv2.boundingRect(label_contour)
        margin = 30 # Aggressive margin to cover torn edges
        
        # Draw black rectangle on mask_objects
        cv2.rectangle(mask_objects, 
                     (max(0, x - margin), max(0, y - margin)), 
                     (min(w, x + w_rect + margin), min(h, y + h_rect + margin)), 
                     0, -1)
        
    # --- NOISE REMOVAL ---
    # Morphological Opening to remove thin scratches/dust
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    total_area_px = 0
    
    img_area = img.shape[0] * img.shape[1]
    max_area_threshold = img_area * 0.10 # 10% of image size

    # Prepare Annotation Image and ROIs
    rois_bytes = []
    
    if output_dir:
        annotated_img = img_corrected.copy()
        # Draw Label Box (Red)
        if label_contour is not None:
             x, y, w_r, h_r = cv2.boundingRect(label_contour)
             # Draw box (Visual only, analysis uses the masked version)
             cv2.rectangle(annotated_img, (x, y), (x+w_r, y+h_r), (0, 0, 255), 3)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter:
        # 1. Tiny dust (< 1mm2 approx 2200 pixels)
        # 2. Scratches (Long/Thin Aspect Ratio)
        # 3. Massive background artifacts (> 10% image)
        
        if area < 2000 or area > max_area_threshold:
            continue
            
        # Aspect Ratio Filter for Scratches
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(max(w, h)) / min(w, h)
        
        # Scratches are usually very long and thin (Ratio > 5 or so)
        # Flowers are roundish or oval (Ratio < 4 usually)
        # Check solidity too? Scratches might be curve.
        # Let's rely on aspect ratio of bounding rect first.
        # If it is small area AND high aspect ratio -> Scratch.
        
        if area < 12000: # Apply shape filter only to small objects
            if aspect_ratio > 4.0:
               # Likely a scratch
               continue
               
            # Check for very thin lines even if curved (Solidity low + High Perimeter/Area ratio)
            # Simple check: Extent (Area / BoundingRectArea)
            extent = float(area) / (w * h)
            # Scratches might have low extent if diagonal, or high if straight.
            # Stick to Aspect Ratio for now as primary filter for linear scratches.
            
        valid_contours.append(cnt)
        total_area_px += area
        
        # Create ROI
        roi_data = roi_writer.create_roi(cnt, img.shape[0], img.shape[1])
        rois_bytes.append((f"{filename}-roi-{i+1}.roi", roi_data))
        
        # Individual Annotation
        if output_dir:
            # Draw Outline (Green)
            cv2.drawContours(annotated_img, [cnt], -1, (0, 255, 0), 2)
            
            # Calculate individual mm2
            ind_mm2 = area * MM2_PER_PIXEL
            
            # Position text near the object
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # Fallback to bounding rect center
                x, y, w, h_rect = cv2.boundingRect(cnt)
                cX, cY = x + w//2, y + h_rect//2
            
            # Draw text (White with black border for visibility)
            text = f"{ind_mm2:.1f}"
            cv2.putText(annotated_img, text, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
            cv2.putText(annotated_img, text, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    total_area_mm2 = total_area_px * MM2_PER_PIXEL

    if output_dir:
        # Draw Total Area at top
        text_str = f"Total Area: {total_area_mm2:.2f} mm2"
        cv2.putText(annotated_img, text_str, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
             
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, annotated_img)
        print(f"Saved {out_path}")
        
    # Save ROIs to Zip
    roi_dir = 'rois'
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)
        
    zip_path = os.path.join(roi_dir, f"{filename}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for roi_name, roi_data in rois_bytes:
            zf.writestr(roi_name, roi_data)
            
    print(f"Saved ROIs to {zip_path}")

    return filename, total_area_px, total_area_mm2

def main():
    output_dir = 'measured_scans'
    output_csv = 'results_final.csv'
    
    # Create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results = []

    # Define source folders and their modes
    folders = [
        {'path': 'scans', 'mode': 'dark'},
        {'path': 'pink_background', 'mode': 'pink'}
    ]
    
    for folder in folders:
        input_dir = folder['path']
        bg_mode = folder['mode']
        
        if not os.path.exists(input_dir):
            print(f"Warning: Directory '{input_dir}' not found. Skipping.")
            continue
            
        print(f"\n--- Processing '{input_dir}' with mode '{bg_mode}' ---")
        
        # Get list of JPG files
        files = sorted(glob.glob(os.path.join(input_dir, '*.JPG')))
        
        if not files:
            print(f"No JPG images found in '{input_dir}'.")
            continue

        # Process images
        for i, filepath in enumerate(files):
            # Always output annotated image
            filename, area_px, area_mm2 = process_image(filepath, output_dir=output_dir, bg_mode=bg_mode)
            if filename:
                results.append([filename, '', '', '', area_px, area_mm2])

    # Save to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Species', 'Sample ID', 'Number of flowers', 'Total Area (pixels)', 'Total Area (mm^2)'])
        writer.writerows(results)
    
    print(f"\nProcessing complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()
