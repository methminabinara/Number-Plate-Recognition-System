import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create structuring element (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Load the main image
image = cv2.imread('assets/images/bike x.jpg')

# Convert the main image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding and Gaussian blur on the main image
_, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
threshold_image = cv2.GaussianBlur(threshold_image, (5, 5), 0)

# Perform morphological opening on the thresholded image
morph_opening = cv2.dilate(threshold_image, kernel, iterations=3)

# Dictionary to represent vehicle types with their templates
vehicle_templates = {
    'Motorbike': ['assets/images/Templates/bike back B1.jpg', 'assets/images/Templates/bike back B2.jpg', 'assets/images/Templates/bike back B3.jpg', 'assets/images/Templates/bike back V1.jpg', 'assets/images/Templates/bike back W1.jpg', 'assets/images/Templates/bike back X1.jpg', 'assets/images/Templates/bike front B1.jpg', 'assets/images/Templates/bike front B2.jpg'],
    'Car': ['assets/images/Templates/car front C1.jpg', 'assets/images/Templates/car back C2.jpg'],
    'Van': ['assets/images/Templates/van front P1.jpg', 'assets/images/Templates/van back P2.jpg'],
    # Add more vehicle types and templates as needed
}

detected_vehicle_type = None

# Loop through vehicle templates and perform template matching
for vehicle_type, template_paths in vehicle_templates.items():
    for template_path in template_paths:
        # Load and process the template image
        template = cv2.imread(template_path)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, thresh_template = cv2.threshold(gray_template, 127, 255, cv2.THRESH_BINARY_INV)

        # Perform template matching
        result = cv2.matchTemplate(morph_opening, thresh_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Check if a good match is found
        if max_val > 0.7:  # Threshold for a good match
            detected_vehicle_type = vehicle_type
            h, w = thresh_template.shape
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            matched_image = cv2.rectangle(image.copy(), top_left, bottom_right, (0, 255, 0), 3)

            print(f"Detected Vehicle Type: {detected_vehicle_type}")
            break  # Break out of the inner loop if a match is found
    if detected_vehicle_type:
        break  # Break out of the outer loop if a match is found

# Display the images using matplotlib
plt.figure(figsize=(10, 8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Grayscale Image
plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Threshold Image
plt.subplot(2, 2, 3)
plt.imshow(threshold_image, cmap='gray')
plt.title('Threshold Image')
plt.axis('off')

# Morphological Opening + Template Matching Result (only if detected)
if detected_vehicle_type:
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Template Matching Result - {detected_vehicle_type}')
    plt.axis('off')

# Show all images
plt.tight_layout()
plt.show()

if not detected_vehicle_type:
    print("No vehicle type detected.")


