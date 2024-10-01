import cv2
import numpy as np
import matplotlib.pyplot as plt

# Desired size for number plate and templates
desired_size = (100, 50)  # Width, Height of the number plate area for template matching

# Create structuring element (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Load the main image
image = cv2.imread('assets/images/bike back B.jpg')

# Convert the main image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding and Gaussian blur on the main image
_, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
threshold_image = cv2.GaussianBlur(threshold_image, (5, 5), 0)

# Perform morphological opening on the thresholded image
morph_opening = cv2.dilate(threshold_image, kernel, iterations=3)

# Find contours and isolate the number plate area
contours, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and assume the largest is the number plate
contour = max(contours, key=cv2.contourArea)

# Get bounding box for the largest contour (number plate)
x, y, w, h = cv2.boundingRect(contour)

# Extract the number plate region and resize it to the desired size
number_plate_region = gray_image[y:y+h, x:x+w]

# Dictionary to represent vehicle types with their templates
vehicle_templates = {
    'Motorbike': ['assets/images/Templates/bike back B1.jpg', 'assets/images/Templates/bike back B2.jpg', 'assets/images/Templates/bike back B3.jpg'],
    'Car': ['assets/images/Templates/car front C1.jpg', 'assets/images/Templates/car back C2.jpg'],
    'Van': ['assets/images/Templates/van front P1.jpg', 'assets/images/Templates/van back P2.jpg'],
    # Add more vehicle types and templates as needed
}

detected_vehicle_type = None
best_match_x_length = float('inf')  # To track the smallest x-direction length for filtering multiple matches

# Loop through vehicle templates and perform template matching
for vehicle_type, template_paths in vehicle_templates.items():
    for template_path in template_paths:
        # Load and process the template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure the template is smaller or equal to the number plate region
        if template.shape[0] > number_plate_region.shape[0] or template.shape[1] > number_plate_region.shape[1]:
            template_resized = cv2.resize(template, (min(template.shape[1], number_plate_region.shape[1]),
                                                     min(template.shape[0], number_plate_region.shape[0])))
        else:
            template_resized = template

        # Apply thresholding to the template
        _, thresh_template = cv2.threshold(template_resized, 127, 255, cv2.THRESH_BINARY_INV)

        # Perform template matching
        result = cv2.matchTemplate(number_plate_region, thresh_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Check if a good match is found and if the x-direction length is smaller
        if max_val > 0.7 and w < best_match_x_length:  # Threshold for a good match and x-direction filtering
            detected_vehicle_type = vehicle_type
            best_match_x_length = w  # Update the best x-direction length
            matched_image = cv2.rectangle(image.copy(), (x, y), (x+w, y+h), (0, 255, 0), 3)

            print(f"Detected Vehicle Type: {detected_vehicle_type} with x-direction length: {w}")

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
