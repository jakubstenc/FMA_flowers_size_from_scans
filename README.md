# Flower Size Analysis and OCR Pipeline

This repository contains a Python-based pipeline for measuring the surface area of flowers from scanned images and preparing the results for manual verification.

## Features

- **Automated Measurements**: Calculates the total area of flowers in each scan.
- **Dual Background Support**:
  - **Dark Background**: Standard processing for flowers on black/dark backgrounds.
  - **Pink Background**: Specialized processing for flowers on pink backgrounds using color subtraction.
- **Label Exclusion**: Automatically detects and masks out the handwritten paper labels to prevent them from being measured as flowers.
- **ImageJ Integration**: Generates `.zip` files containing standard ImageJ ROIs (Region of Interest) for every detected flower, allowing for easy manual correction.
- **Visual Validation**: Outputs annotated images showing the detected contours and individual measurements.

## Installation

Ensure you have Python 3 installed. The following dependencies are required:

```bash
pip install opencv-python numpy
```

## Usage

1.  **Organize your scans**:
    - Place dark background scans in a folder named `scans`.
    - Place pink background scans in a folder named `pink_background`.

2.  **Run the analysis**:
    ```bash
    python3 measure_flowers.py
    ```

3.  **Review Results**:
    - **Data**: Check `results_final.csv` for the compiled list of files and measurements.
    - **Visuals**: Check the `measured_scans/` folder to see the red box (label) and green outlines (flowers).
    - **Manual Edit**: Use the generated zip files in the `rois/` folder to load outlines into FIJI/ImageJ.

## The Algorithm

The pipeline follows these steps for each image:

1.  **Preprocessing**:
    - Applies gamma correction (0.5) to brighten the image and reveal details in dark areas.
    - Converts the image to the HSV color space for better color segmentation.

2.  **Label Detection**:
    - **Dark Mode**: Detects the label as a large, high-value (bright) region.
    - **Pink Mode**: Detects the label as a white object (High Value, very Low Saturation) to distinguish it from the saturated pink background.
    - The detected label region is masked out (ignored) during flower detection.

3.  **Flower Detection**:
    - **Dark Mode**: Thresholds the Value channel (V > 60) to find all objects that are not the dark background.
    - **Pink Mode**: Masks out the specific range of Pink hues. Everything "Not Pink" (and not the white label) is considered a flower.

4.  **Filtering & measurement**:
    - Removes small noise (dust) and scratch-like artifacts based on aspect ratio.
    - Calculates the area in pixels and converts to $mm^2$ using a fixed DPI of 1200.

5.  **Output Generation**:
    - **CSV**: Filename, Placeholder columns for manual entry (Species, ID, Count), and Total Area.
    - **ROIs**: A custom binary writer generates ImageJ-compatible `.roi` files, zipped together for each image.

## Manual Verification (FIJI/ImageJ)

If you suspect a measurement is incorrect, you can manually verify and fix it using FIJI/ImageJ.

**Video Tutorial:**
[Click here to watch the screencast (roi_verification_screencast.webm)](roi_verification_screencast.webm)

### Steps to Edit and Re-measure:
1.  **Open Image**: Drag the original image in FIJI.
2.  **Open ROI Manager**: `Analyze > Tools > ROI Manager`.
3.  **Load ROIs**: Drag and drop the corresponding `IMAGxxxx.JPG.zip` file from the `rois/` folder onto the ROI Manager window.
4.  **Edit**:
    -   **Delete**: Select incorrect outlines in the list and press `Delete`.
    -   **Add**: Use the Freehand or Polygon selection tool to trace a missing flower, then press `t` (or clicks `Add` in ROI Manager) to add it to the list.
5.  **Re-Measure**:
    -   In ROI Manager, Click **Deselect** to ensure no specific ROI is highlighted (this ensures *all* ROIs are processed).
    -   Click the **Measure** button in the ROI Manager.
    -   A "Results" table will pop up.
    -   **Total Area**: In the Results table, check the `Area` column. You can copy these values to Excel/CSV to sum them up. 
    -   *Note*: Ensure your image scale is set correctly (Analyze > Set Scale) if you want real units, otherwise these are typically pixels unless the scanner metadata is read. The script assumes 1200 DPI. To match the script's mm², you'd need to set the scale in FIJI to `47.244 pixels/mm` (1200/25.4).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
