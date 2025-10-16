import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

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


def detect_highlights(image, skin_mask):
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
    
    # Morphological operations to refine the highlight mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    
    return highlight_mask


def inpaint_highlights(image, highlight_mask):
    """
    Remove highlights using inpainting technique.
    This blends the highlighted areas with surrounding skin tone.
    """
    # Use OpenCV's inpainting algorithm
    # INPAINT_TELEA works well for small regions
    result = cv2.inpaint(image, highlight_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    return result


def remove_reflections_advanced(image, highlight_mask):
    """
    Advanced method to remove reflections by blending with local skin tone.
    """
    result = image.copy().astype(np.float32)
    
    # Dilate the mask to get surrounding skin pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_mask = cv2.dilate(highlight_mask, kernel, iterations=2)
    
    # Get the border region (surrounding skin)
    border_mask = cv2.subtract(dilated_mask, highlight_mask)
    
    # For each color channel
    for c in range(3):
        channel = result[:, :, c]
        
        # Calculate local mean of surrounding skin
        border_pixels = channel[border_mask > 0]
        if len(border_pixels) > 0:
            local_mean = np.median(border_pixels)
            
            # Blend highlight regions towards local skin tone
            highlight_region = highlight_mask > 0
            channel[highlight_region] = local_mean * 0.7 + channel[highlight_region] * 0.3
    
    # Apply bilateral filter to smooth while preserving edges
    result = result.astype(np.uint8)
    result = cv2.bilateralFilter(result, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Blend the result back using the highlight mask
    mask_3d = cv2.cvtColor(highlight_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    
    # Smooth the mask edges for better blending
    mask_3d = cv2.GaussianBlur(mask_3d, (15, 15), 0)
    
    # Blend original and processed image
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
    
    # Step 1: Detect skin regions
    print("Detecting skin regions...")
    skin_mask = detect_skin(image)
    
    # Step 2: Detect highlights on skin
    print("Detecting highlights/reflections...")
    highlight_mask = detect_highlights(image, skin_mask)
    
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
    print("Saved intermediate masks: skin_mask.png, highlight_mask.png")
    
    print("Done!")


if __name__ == "__main__":
    # Process the input photo
    input_photo = "input_photo.jpeg"
    output_photo = "output_photo.jpeg"
    
    # You can choose between 'inpaint' (faster, good for small highlights)
    # or 'advanced' (better for larger reflections)
    process_image(input_photo, output_photo, method='inpaint')
    
    print(f"\nOutput saved to: {output_photo}")
    print("Intermediate masks saved for inspection:")
    print("  - skin_mask.png: Shows detected skin regions")
    print("  - highlight_mask.png: Shows detected reflections")
