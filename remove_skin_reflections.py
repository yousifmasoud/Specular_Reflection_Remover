import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def detect_eye_region_precise(image, eye_x, eye_y, eye_w, eye_h):
    """
    Precisely detect the actual eye area (iris, pupil, sclera) within a detected eye bounding box.
    Returns a refined mask for just the eye structure.
    """
    # Extract the eye region
    eye_roi = image[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
    
    if eye_roi.size == 0:
        return np.zeros((eye_h, eye_w), dtype=np.uint8)
    
    # Convert to grayscale and HSV for analysis
    eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    eye_hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(eye_hsv)
    
    # Create refined mask
    refined_mask = np.zeros((eye_h, eye_w), dtype=np.uint8)
    
    # Method 1: Detect dark regions (pupil and iris)
    # Pupil and iris are typically darker than surrounding skin
    _, dark_mask = cv2.threshold(eye_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 2: Detect eye whites (sclera) - low saturation, medium-high brightness
    sclera_mask = ((s < 40) & (v > 100) & (v < 240)).astype(np.uint8) * 255
    
    # Method 3: Detect iris region - has some color saturation
    iris_mask = ((s > 15) & (v > 30) & (v < 200)).astype(np.uint8) * 255
    
    # Combine masks: we want dark areas (pupil/iris) OR white areas (sclera)
    eye_structure = cv2.bitwise_or(dark_mask, sclera_mask)
    eye_structure = cv2.bitwise_or(eye_structure, iris_mask)
    
    # Clean up the mask - remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eye_structure = cv2.morphologyEx(eye_structure, cv2.MORPH_CLOSE, kernel, iterations=2)
    eye_structure = cv2.morphologyEx(eye_structure, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find the largest connected component (the actual eye)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eye_structure, connectivity=8)
    
    if num_labels > 1:
        # Get the largest component (excluding background)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        refined_mask = (labels == largest_component).astype(np.uint8) * 255
    else:
        refined_mask = eye_structure
    
    # Apply edge detection to find eye boundaries
    edges = cv2.Canny(eye_gray, 30, 100)
    
    # Dilate edges slightly to create boundary
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Fill the region inside the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find contour that's most likely the eye (centered, reasonable size)
        eye_center_x, eye_center_y = eye_w // 2, eye_h // 2
        
        best_contour = None
        best_score = float('inf')
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Eye should be a reasonable portion of the detection box
            if area < (eye_w * eye_h * 0.1) or area > (eye_w * eye_h * 0.8):
                continue
            
            # Calculate how centered the contour is
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = np.sqrt((cx - eye_center_x)**2 + (cy - eye_center_y)**2)
                
                if dist < best_score:
                    best_score = dist
                    best_contour = contour
        
        if best_contour is not None:
            # Create mask from best contour
            contour_mask = np.zeros((eye_h, eye_w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [best_contour], -1, 255, -1)
            
            # Combine with color-based detection
            refined_mask = cv2.bitwise_or(refined_mask, contour_mask)
    
    # Create an almond/eye shape mask as a constraint
    # Eyes are typically almond-shaped, not circular
    almond_mask = np.zeros((eye_h, eye_w), dtype=np.uint8)
    center = (eye_w // 2, eye_h // 2)
    
    # Create almond shape (ellipse with narrower width at top and bottom)
    axes = (int(eye_w * 0.45), int(eye_h * 0.35))
    cv2.ellipse(almond_mask, center, axes, 0, 0, 360, 255, -1)
    
    # Intersect with almond shape to ensure natural eye shape
    refined_mask = cv2.bitwise_and(refined_mask, almond_mask)
    
    # Final smoothing
    refined_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
    
    return refined_mask


def detect_eyes(image):
    """
    Detect eyes in the image and return a precise mask of actual eye areas.
    Returns a binary mask where eye regions are white (255).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade for eye detection
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    # Also try face detection to better locate eye regions
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Create empty mask
    eye_mask = np.zeros(gray.shape, dtype=np.uint8)
    
    # Detect faces first to narrow down search area
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        # Search for eyes within face regions
        for (x, y, w, h) in faces:
            # Define eye region (upper half of face)
            roi_gray = gray[y:y+int(h*0.6), x:x+w]
            roi_x, roi_y = x, y
            
            # Detect eyes in this region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
            
            for (ex, ey, ew, eh) in eyes:
                # Calculate absolute coordinates
                abs_x = roi_x + ex
                abs_y = roi_y + ey
                abs_w = ew
                abs_h = eh
                
                # Ensure within image bounds
                abs_x = max(0, abs_x)
                abs_y = max(0, abs_y)
                abs_w = min(image.shape[1] - abs_x, abs_w)
                abs_h = min(image.shape[0] - abs_y, abs_h)
                
                # Get precise eye mask for this region
                precise_mask = detect_eye_region_precise(image, abs_x, abs_y, abs_w, abs_h)
                
                # Place the precise mask in the full image mask
                eye_mask[abs_y:abs_y+abs_h, abs_x:abs_x+abs_w] = cv2.bitwise_or(
                    eye_mask[abs_y:abs_y+abs_h, abs_x:abs_x+abs_w],
                    precise_mask
                )
    else:
        # No face detected, try to detect eyes directly in whole image
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        for (ex, ey, ew, eh) in eyes:
            # Ensure within bounds
            ex = max(0, ex)
            ey = max(0, ey)
            ew = min(image.shape[1] - ex, ew)
            eh = min(image.shape[0] - ey, eh)
            
            # Get precise eye mask for this region
            precise_mask = detect_eye_region_precise(image, ex, ey, ew, eh)
            
            # Place the precise mask in the full image mask
            eye_mask[ey:ey+eh, ex:ex+ew] = cv2.bitwise_or(
                eye_mask[ey:ey+eh, ex:ex+ew],
                precise_mask
            )
    
    # Apply minimal smoothing to final mask
    if np.sum(eye_mask) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eye_mask = cv2.morphologyEx(eye_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        eye_mask = cv2.GaussianBlur(eye_mask, (5, 5), 0)
    
    return eye_mask


def detect_skin(image):
    """
    Detect skin regions in the image using multiple color spaces with refined masking.
    Returns a binary mask where skin regions are white (255).
    """
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # HSV skin detection
    # Typical skin tone ranges in HSV
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 170, 255], dtype=np.uint8)
    skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # YCrCb skin detection (more robust)
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # RGB-based skin detection for additional refinement
    b, g, r = cv2.split(image)
    rgb_mask = ((r > 95) & (g > 40) & (b > 20) & 
                (r > g) & (r > b) & 
                (np.abs(r.astype(int) - g.astype(int)) > 15)).astype(np.uint8) * 255
    
    # Combine all masks using weighted approach
    skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)
    skin_mask = cv2.bitwise_and(skin_mask, rgb_mask)
    
    # Advanced morphological operations for refinement
    # Remove small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Close gaps
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
    
    # Remove small isolated regions using connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skin_mask, connectivity=8)
    
    # Calculate minimum area threshold (remove components smaller than 1% of image area)
    min_area = (image.shape[0] * image.shape[1]) * 0.01
    
    # Create refined mask keeping only large components
    refined_mask = np.zeros_like(skin_mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            refined_mask[labels == i] = 255
    
    # Fill holes in the mask
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    # Smooth edges with bilateral filter to preserve boundaries
    refined_mask = cv2.bilateralFilter(refined_mask, 9, 75, 75)
    
    # Final Gaussian blur for soft edges
    refined_mask = cv2.GaussianBlur(refined_mask, (7, 7), 0)
    
    # Apply edge-preserving smoothing using morphological gradient
    gradient = cv2.morphologyEx(refined_mask, cv2.MORPH_GRADIENT, kernel_small)
    refined_mask = cv2.subtract(refined_mask, gradient // 2)
    
    return refined_mask


def detect_highlights(image, skin_mask, eye_mask=None):
    """
    Detect bright highlights/reflections on skin areas.
    Returns a mask of the highlight regions.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert to HSV for saturation and value analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Detect highlights: high brightness (V channel) and low saturation
    # Highlights are typically very bright and desaturated
    highlight_mask = np.zeros_like(gray)
    
    # Threshold for brightness (adjust these values if needed)
    brightness_threshold = 200  # V channel threshold
    saturation_threshold = 100   # S channel threshold (low saturation)
    
    # Create highlight mask based on high brightness and low saturation
    highlight_mask = ((v > brightness_threshold) & (s < saturation_threshold)).astype(np.uint8) * 255
    
    # Only keep highlights that are on skin
    highlight_mask = cv2.bitwise_and(highlight_mask, skin_mask)
    
    # Exclude eye regions from highlight mask
    if eye_mask is not None and np.sum(eye_mask) > 0:
        # Create inverse eye mask (everything except eyes)
        eye_mask_binary = (eye_mask > 127).astype(np.uint8) * 255
        inverse_eye_mask = cv2.bitwise_not(eye_mask_binary)
        highlight_mask = cv2.bitwise_and(highlight_mask, inverse_eye_mask)
    
    # Morphological operations to refine the highlight mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    
    return highlight_mask


def inpaint_highlights(image, highlight_mask):
    """
    Remove highlights using enhanced inpainting technique.
    This blends the highlighted areas with surrounding skin tone.
    """
    # Use a two-stage inpainting approach
    # First stage: Navier-Stokes based inpainting for structure
    result = cv2.inpaint(image, highlight_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
    
    # Second stage: Apply local skin tone correction
    result = blend_with_local_skin_tone(image, result, highlight_mask)
    
    return result


def blend_with_local_skin_tone(original, inpainted, mask):
    """
    Enhanced blending that matches local skin tone characteristics.
    """
    result = inpainted.copy().astype(np.float32)
    original_float = original.astype(np.float32)
    
    # Create a ring around each highlight to sample surrounding skin
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    
    # Erode mask slightly to avoid highlight pixels
    eroded_mask = cv2.erode(mask, kernel_inner, iterations=1)
    
    # Dilate to get surrounding region
    dilated_mask = cv2.dilate(eroded_mask, kernel_outer, iterations=1)
    
    # Border region = surrounding skin
    border_mask = cv2.subtract(dilated_mask, eroded_mask)
    
    # Get connected components of highlights to process each separately
    num_labels, labels = cv2.connectedComponents(mask)
    
    for label in range(1, num_labels):
        # Get current highlight region
        current_highlight = (labels == label).astype(np.uint8) * 255
        
        # Get local border for this specific highlight
        dilated_current = cv2.dilate(current_highlight, kernel_outer, iterations=1)
        local_border = cv2.bitwise_and(border_mask, dilated_current)
        
        if np.sum(local_border) == 0:
            continue
        
        # Sample surrounding skin colors
        for c in range(3):
            channel = result[:, :, c]
            orig_channel = original_float[:, :, c]
            
            # Get surrounding skin pixels
            border_pixels = orig_channel[local_border > 0]
            
            if len(border_pixels) > 10:
                # Use percentile to avoid outliers
                local_median = np.percentile(border_pixels, 50)
                local_q25 = np.percentile(border_pixels, 25)
                local_q75 = np.percentile(border_pixels, 75)
                
                # Get inpainted pixels in this region
                highlight_pixels = channel[current_highlight > 0]
                
                # Calculate mean of inpainted region
                inpainted_mean = np.mean(highlight_pixels)
                
                # Adjust to match local skin tone distribution
                if inpainted_mean > local_median:
                    # If too bright, darken towards median
                    scale_factor = local_median / (inpainted_mean + 1e-6)
                    channel[current_highlight > 0] = np.clip(
                        highlight_pixels * scale_factor * 0.85 + local_median * 0.15,
                        local_q25, local_q75
                    )
                else:
                    # Already darker, minor adjustment
                    channel[current_highlight > 0] = np.clip(
                        highlight_pixels * 0.7 + local_median * 0.3,
                        local_q25, local_q75
                    )
    
    result = result.astype(np.uint8)
    
    # Apply edge-preserving smoothing only on corrected regions
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    smoothed = cv2.bilateralFilter(result, d=9, sigmaColor=50, sigmaSpace=50)
    
    # Blend smoothed version only in highlight regions
    mask_float = mask_3d.astype(np.float32) / 255.0
    result = (smoothed * mask_float + result * (1 - mask_float)).astype(np.uint8)
    
    return result


def remove_reflections_advanced(image, highlight_mask):
    """
    Advanced method to remove reflections by intelligent blending with local skin tone.
    Uses texture synthesis and color matching for natural results.
    """
    result = image.copy().astype(np.float32)
    original = image.astype(np.float32)
    
    # Create sampling rings around highlights
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    
    # Shrink mask slightly to ensure we're working with highlight core
    core_mask = cv2.erode(highlight_mask, kernel_small, iterations=1)
    
    # Create rings for sampling at different distances
    ring1 = cv2.dilate(core_mask, kernel_medium, iterations=1)
    ring1 = cv2.subtract(ring1, core_mask)
    
    ring2 = cv2.dilate(core_mask, kernel_large, iterations=1)
    ring2 = cv2.subtract(ring2, cv2.dilate(core_mask, kernel_medium, iterations=1))
    
    # Combine rings with weights (closer ring has more influence)
    sampling_mask = cv2.addWeighted(ring1, 0.7, ring2, 0.3, 0)
    
    # Process each connected highlight region independently
    num_labels, labels = cv2.connectedComponents(highlight_mask)
    
    for label in range(1, num_labels):
        current_highlight = (labels == label).astype(np.uint8) * 255
        
        # Get local sampling region for this highlight
        dilated_current = cv2.dilate(current_highlight, kernel_large, iterations=1)
        local_sampling = cv2.bitwise_and(sampling_mask, dilated_current)
        
        if np.sum(local_sampling) < 100:  # Not enough samples
            continue
        
        # Extract color statistics from surrounding skin
        b_samples = original[:, :, 0][local_sampling > 0]
        g_samples = original[:, :, 1][local_sampling > 0]
        r_samples = original[:, :, 2][local_sampling > 0]
        
        # Calculate robust statistics (avoid outliers)
        b_target = np.percentile(b_samples, 50)
        g_target = np.percentile(g_samples, 50)
        r_target = np.percentile(r_samples, 50)
        
        b_std = np.std(b_samples)
        g_std = np.std(g_samples)
        r_std = np.std(r_samples)
        
        # Create distance transform for smooth blending from edges
        dist_transform = cv2.distanceTransform(current_highlight, cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
        
        # Apply Gaussian blur to distance for smoother transition
        dist_transform = cv2.GaussianBlur(dist_transform, (11, 11), 0)
        
        # Get current highlight pixels
        highlight_coords = current_highlight > 0
        
        # Blend towards target skin tone with distance-based weighting
        for c, target, std_dev in [(0, b_target, b_std), (1, g_target, g_std), (2, r_target, r_std)]:
            channel = result[:, :, c]
            current_pixels = channel[highlight_coords]
            dist_values = dist_transform[highlight_coords]
            
            # Stronger correction in center, gentler at edges
            blend_strength = 0.6 + 0.3 * dist_values  # 0.6 to 0.9
            
            # Add some texture variation
            noise = np.random.normal(0, std_dev * 0.1, size=current_pixels.shape)
            
            # Blend with target color
            new_values = (current_pixels * (1 - blend_strength) + 
                         target * blend_strength + noise)
            
            # Clamp to valid range and reasonable bounds
            new_values = np.clip(new_values, 
                               max(0, target - 2 * std_dev), 
                               min(255, target + 2 * std_dev))
            
            channel[highlight_coords] = new_values
    
    result = result.astype(np.uint8)
    
    # Multi-scale smoothing for natural texture
    # Fine details
    fine = cv2.bilateralFilter(result, d=5, sigmaColor=30, sigmaSpace=30)
    # Medium details  
    medium = cv2.bilateralFilter(result, d=9, sigmaColor=50, sigmaSpace=50)
    
    # Blend scales based on mask
    mask_3d = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    mask_3d = cv2.GaussianBlur(mask_3d, (21, 21), 0)
    
    # Use fine smoothing in center, medium at edges
    result = (fine * 0.4 + medium * 0.6).astype(np.uint8)
    
    # Final blend with original using smooth mask
    final_result = (result * mask_3d + image * (1 - mask_3d)).astype(np.uint8)
    
    return final_result


def process_image(input_path, output_path, method='advanced'):
    """
    Main function to process the image and remove skin reflections.
    
    Parameters:
    - input_path: path to input image
    - output_path: path to save output image
    - method: 'inpaint' or 'advanced' (default: 'inpaint')
    """
    # Read the input image
    print("Reading input image...")
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"Error: Could not read image from {input_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Step 1: Detect and exclude eyes
    print("Detecting eyes to exclude from processing...")
    eye_mask = detect_eyes(image)
    if np.sum(eye_mask) > 0:
        print(f"Detected and will exclude {np.sum(eye_mask > 0)} eye region pixels")
    else:
        print("No eyes detected, processing entire face")
    
    # Step 2: Detect skin regions
    print("Detecting skin regions...")
    skin_mask = detect_skin(image)
    
    # Exclude eyes from skin mask
    if np.sum(eye_mask) > 0:
        eye_mask_binary = (eye_mask > 127).astype(np.uint8) * 255
        inverse_eye_mask = cv2.bitwise_not(eye_mask_binary)
        skin_mask = cv2.bitwise_and(skin_mask, inverse_eye_mask)
    
    # Step 3: Detect highlights on skin (excluding eyes)
    print("Detecting highlights/reflections...")
    highlight_mask = detect_highlights(image, skin_mask, eye_mask)
    
    # Check if any highlights were detected
    if np.sum(highlight_mask) == 0:
        print("No significant highlights detected. Saving original image.")
        cv2.imwrite(output_path, image)
        return
    
    print(f"Detected {np.sum(highlight_mask > 0)} highlight pixels")
    
    # Step 3: Remove highlights
    print(f"Removing reflections using {method} method...")
    if method == 'inpaint':
        result = inpaint_highlights(image, highlight_mask)
    elif method == 'advanced':
        result = remove_reflections_advanced(image, highlight_mask)
    else:
        print(f"Unknown method: {method}. Using inpaint.")
        result = inpaint_highlights(image, highlight_mask)
    
    # Save the result
    print(f"Saving result to {output_path}...")
    cv2.imwrite(output_path, result)
    
    # Optionally save intermediate masks for debugging
    cv2.imwrite('skin_mask.png', skin_mask)
    cv2.imwrite('highlight_mask.png', highlight_mask)
    if np.sum(eye_mask) > 0:
        cv2.imwrite('eye_mask.png', eye_mask)
        print("Saved intermediate masks: skin_mask.png, highlight_mask.png, eye_mask.png")
    else:
        print("Saved intermediate masks: skin_mask.png, highlight_mask.png")
    
    print("Done!")


if __name__ == "__main__":
    # Process the input photo
    input_photo = "input_photo.jpeg"
    output_photo = "output_photo.jpeg"
    
    # You can choose between 'inpaint' (enhanced inpainting with skin tone matching)
    # or 'advanced' (intelligent texture synthesis and color blending - RECOMMENDED)
    process_image(input_photo, output_photo, method='advanced')
    
    print(f"\nOutput saved to: {output_photo}")
    print("Intermediate masks saved for inspection:")
    print("  - skin_mask.png: Shows detected skin regions (eyes excluded)")
    print("  - highlight_mask.png: Shows detected reflections (eyes excluded)")
    print("  - eye_mask.png: Shows detected eye regions (if any eyes found)")
