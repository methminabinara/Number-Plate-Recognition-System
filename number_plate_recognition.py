import cv2

# Read the input image
input_image = cv2.imread("F:/University/3000 Level/Semester 1/Computer Science/CSC 3141/Project/Project/images/discover150.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Apply Gaussian blur for noise reduction
blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

# Resize images to fit within the screen dimensions
height, width = input_image.shape[:2]
max_height = 800
max_width = 1200
if height > max_height or width > max_width:
    scale_factor = min(max_height/height, max_width/width)
    input_image = cv2.resize(input_image, None, fx=scale_factor, fy=scale_factor)
    gray_image = cv2.resize(gray_image, None, fx=scale_factor, fy=scale_factor)
    binary_image = cv2.resize(binary_image, None, fx=scale_factor, fy=scale_factor)
    blurred_image = cv2.resize(blurred_image, None, fx=scale_factor, fy=scale_factor)

# Display the original and processed images
cv2.imshow("Original Image", input_image)
cv2.imshow("Grayscale Image", gray_image)
cv2.imshow("Binary Image", binary_image)
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
