import struct
import numpy as np

def create_roi(contour, img_height, img_width, roi_name=None):
    """
    Creates an ImageJ binary ROI (Polygon type) from a contour.
    
    Args:
        contour: Numpy array of point coordinates (N, 1, 2) or (N, 2).
        img_height: Height of the image (needed for bounding box checks).
        img_width: Width of the image.
        roi_name: Name of the ROI (optional).
    
    Returns:
        bytes: The binary data of the .roi file.
    """
    
    # Flatten contour if needed
    if len(contour.shape) == 3:
        pts = contour[:, 0, :]
    else:
        pts = contour

    n_points = len(pts)
    
    # Calculate Bounding Box
    x = pts[:, 0]
    y = pts[:, 1]
    
    top = int(np.min(y))
    left = int(np.min(x))
    bottom = int(np.max(y))
    right = int(np.max(x))
    
    # Coordinates relative to bounding box
    x_rel = x - left
    y_rel = y - top
    
    # Header Format (Big Endian)
    # 0-3: "Iout"
    # 4-5: Version (217)
    # 6-7: Roi Type (0=Polygon, 1=Rect, 2=Oval, 3=Line, 4=FreeLine, 7=Freehand, 9=Point)
    #      We use 0 (Polygon) or 4 (FreeLine) or 7 (Freehand/Traced). 
    #      0 (Polygon) is standard for closed shapes. 7 (Freehand) is also common.
    # 8-9: Top
    # 10-11: Left
    # 12-13: Bottom
    # 14-15: Right
    # 16-17: NCoordinates
    
    roi_type = 0 # Polygon
    version = 217
    
    header = struct.pack('>4s2h4hH', 
                         b'Iout', 
                         version, roi_type, 
                         top, left, bottom, right, 
                         n_points)
    
    # Pad header to 64 bytes
    header += b'\x00' * (64 - len(header))
    
    # Coordinate Data (Short) relative to (left, top)
    coords_x = struct.pack(f'>{n_points}h', *x_rel)
    coords_y = struct.pack(f'>{n_points}h', *y_rel)
    
    data = header + coords_x + coords_y
    
    # Optional: Roi Name
    # Name is stored after coordinates, but the format is tricky with 
    # specific headers/subtype flags for extended data.
    # Simple .roi files often just have the data. 
    # Let's keep it simple for now to avoid corruption.
    # Usually filenames in the zip act as names in ROI Manager.
    
    return data
