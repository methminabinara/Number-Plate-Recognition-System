import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create structuring element (kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Load the main image and the template image
image = cv2.imread('assets/images/bike back B.jpg')
template = cv2.imread('assets/images/Templates/bike back B1.jpg')

# Convert both images to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
thresh_template = cv2.threshold(gray_template,127,255,cv2.THRESH_BINARY_INV)

# Apply thresholding and Gaussian blur on the main image
_, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
threshold_image = cv2.GaussianBlur(threshold_image, (5, 5), 0)

# Perform morphological opening
morph_opening = cv2.dilate(threshold_image, kernel, iterations=3)

# Perform template matching
result = cv2.matchTemplate(morph_opening, thresh_template, cv2.TM_CCOEFF_NORMED)

# Get the location of the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Get the size of the template
h, w = gray_template.shape

# Draw a rectangle around the best match
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
matched_image = cv2.rectangle(image.copy(), top_left, bottom_right, (0, 255, 0), 3)

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

# Morphological Opening Image
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title('Template Matching Result')
plt.axis('off')

# Show all images
plt.tight_layout()
plt.show()
