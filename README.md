# Skin Reflection Removal Tool

This Python script automatically detects and removes white light reflections from human skin in photos while preserving the background.

## Features

- **Eye Detection & Exclusion**: Automatically detects and excludes eye regions from processing to preserve natural eye highlights and reflections
- **Refined Skin Detection**: Uses HSV, YCrCb, and RGB color spaces with advanced morphological operations for accurate skin region detection
- **Highlight Detection**: Identifies bright reflections based on brightness and saturation, excluding eyes
- **Smart Removal**: Two methods available:
  - `inpaint`: Enhanced inpainting with local skin tone matching
  - `advanced`: Intelligent texture synthesis and color blending (RECOMMENDED)
- **Background Preservation**: Only processes skin areas, leaving background completely untouched
- **Natural Blending**: Multi-scale smoothing and distance-based blending for seamless results
- **Debug Output**: Saves intermediate masks for inspection (skin, highlights, and eyes)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input image as `input_photo.jpeg` in the same directory as the script

2. Run the script:
```bash
python remove_skin_reflections.py
```

3. The processed image will be saved as `output_photo.jpeg`

### Using Different Methods

Edit the script to choose between methods:

```python
# Fast inpainting method (default)
process_image(input_photo, output_photo, method='inpaint')

# Advanced blending method (better for larger reflections)
process_image(input_photo, output_photo, method='advanced')
```

### Custom Input/Output Files

You can modify the file paths in the script:

```python
input_photo = "your_input_image.jpg"
output_photo = "your_output_image.jpg"
process_image(input_photo, output_photo, method='inpaint')
```

## How It Works

1. **Eye Detection**: Uses Haar Cascade classifiers to detect and exclude eye regions from processing
2. **Refined Skin Detection**: Identifies skin regions using multi-color space analysis (HSV, YCrCb, RGB) with:
   - Connected components analysis to remove false positives
   - Advanced morphological operations for refinement
   - Automatic eye region exclusion
3. **Highlight Detection**: Finds bright, desaturated areas within skin regions (excluding eyes)
4. **Intelligent Blending**: Removes highlights using:
   - **Inpaint method**: Enhanced Navier-Stokes inpainting with local skin tone matching
   - **Advanced method**: Texture synthesis with per-highlight color adaptation, distance-based blending, and multi-scale smoothing
5. **Background Preservation**: Only skin regions (excluding eyes) are processed; everything else remains unchanged

## Output Files

- `output_photo.jpeg`: Final processed image
- `skin_mask.png`: Shows detected skin regions with eyes excluded (for debugging)
- `highlight_mask.png`: Shows detected reflections with eyes excluded (for debugging)
- `eye_mask.png`: Shows detected eye regions (for debugging, if eyes are found)

## Adjusting Sensitivity

You can fine-tune detection parameters in the script:

### Skin Detection
In `detect_skin()` function:
```python
# Adjust HSV ranges
lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
upper_hsv = np.array([20, 170, 255], dtype=np.uint8)
```

### Highlight Detection
In `detect_highlights()` function:
```python
brightness_threshold = 200  # Lower to detect dimmer highlights
saturation_threshold = 50   # Higher to be more selective
```

## Troubleshooting

- **No highlights detected**: Lower the `brightness_threshold` value
- **Too much removed**: Increase `brightness_threshold` or decrease `saturation_threshold`
- **Skin not detected**: Adjust skin color ranges in `detect_skin()`
- **Edges look harsh**: Increase blur parameters in the blending section

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- SciPy

## License

Free to use and modify.
