# import cv2

# # Read the input image
# input_image = cv2.imread("F:/University/3000 Level/Semester 1/Computer Science/CSC 3141/Project/Project/images/discover150.jpg")

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to create a binary image
# _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# # Apply Gaussian blur for noise reduction
# blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

# # Resize images to fit within the screen dimensions
# height, width = input_image.shape[:2]
# max_height = 800
# max_width = 1200
# if height > max_height or width > max_width:
#     scale_factor = min(max_height/height, max_width/width)
#     input_image = cv2.resize(input_image, None, fx=scale_factor, fy=scale_factor)
#     gray_image = cv2.resize(gray_image, None, fx=scale_factor, fy=scale_factor)
#     binary_image = cv2.resize(binary_image, None, fx=scale_factor, fy=scale_factor)
#     blurred_image = cv2.resize(blurred_image, None, fx=scale_factor, fy=scale_factor)

# # Display the original and processed images
# cv2.imshow("Original Image", input_image)
# cv2.imshow("Grayscale Image", gray_image)
# cv2.imshow("Binary Image", binary_image)
# cv2.imshow("Blurred Image", blurred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




import cv2
import numpy as np

# Function to preprocess the input image (grayscale, thresholding, noise reduction)
def preprocess_image(image_path):
    # Read the input image
    input_image = cv2.imread("F:/University/3000 Level/Semester 1/Computer Science/CSC 3141/Project/Number-Plate-Recognition-System/assets/images/bike back B.jpg")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Apply Gaussian blur for noise reduction
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

    return input_image, blurred_image

# Function to detect if the number plate is front (1 row) or back (2 rows)
def detect_plate_type(image):
    # Simple heuristic: Check image height and width ratio to determine rows
    height, width = image.shape[:2]
    
    if height / width > 0.3:  # Assumes back plate has larger height relative to width
        return "back"
    return "front"

# Function for template matching (template comparison)
def template_matching(plate_image, templates):
    match_results = {}
    for label, template_path in templates.items():
        template = cv2.imread(template_path, 0)
        template = cv2.GaussianBlur(template, (5, 5), 0)  # Same preprocessing as the plate image

        # Perform template matching
        result = cv2.matchTemplate(plate_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Save the result for each template
        match_results[label] = max_val
    
    # Return the label with the highest match value
    best_match = max(match_results, key=match_results.get)
    return best_match

# Function to crop the first letter (coordinates vary for front and back plates)
def crop_first_letter(image, plate_type):
    if plate_type == "front":
        x, y, w, h = 60, 10, 80, 150  # Adjust these values for front plates
    else:  # back plate (two rows)
        x, y, w, h = 40, 60, 80, 120  # Adjust these values for back plates
    
    return image[y:y+h, x:x+w]

# Main code execution
if __name__ == "__main__":
    # Paths to the templates
    templates = {
        'A': "templates/A.jpg",
        'B': "templates/B.jpg",
        'C': "templates/C.jpg",
        'G': "templates/G.jpg",
        'H': "templates/H.jpg",
        'J': "templates/J.jpg",
        'K': "templates/K.jpg",
        'M': "templates/M.jpg",
        'N': "templates/N.jpg",
        'P': "templates/P.jpg",
        'Q': "templates/Q.jpg",
        'T': "templates/T.jpg",
        'U': "templates/U.jpg",
        'V': "templates/V.jpg",
        'W': "templates/W.jpg",
        'X': "templates/X.jpg",
        'Y': "templates/Y.jpg"
    }

    # Step 1: Preprocess the input image
    image_path = "path_to_your_image.jpg"
    input_image, processed_image = preprocess_image(image_path)

    # Step 2: Detect if it's a front or back number plate
    plate_type = detect_plate_type(processed_image)
    print(f"Detected plate type: {plate_type}")

    # Step 3: Crop the region of the first letter
    first_letter_roi = crop_first_letter(processed_image, plate_type)

    # Step 4: Perform template matching
    best_match = template_matching(first_letter_roi, templates)

    # Vehicle types associated with each letter
    vehicle_types = {
        'A': "Three-wheel",
        'B': "Motor bicycle",
        'C': "Motor car",
        'G': "Quadricycle",
        'H': "All vehicle types",
        'J': "All vehicle types",
        'K': "Motor car",
        'M': "Motor bicycle",
        'N': "Bus",
        'P': "Van",
        'Q': "Three-wheel",
        'T': "Motor bicycle",
        'U': "Motor bicycle",
        'V': "Motor bicycle",
        'W': "Motor bicycle",
        'X': "Motor bicycle",
        'Y': "Three-wheel"
    }

    # Step 5: Display the vehicle type based on the first letter
    vehicle_type = vehicle_types.get(best_match, "Unknown")
    print(f"Best match: {best_match} - Vehicle type: {vehicle_type}")

    # Step 6: Resize images to fit within the screen dimensions (optional for display)
    height, width = input_image.shape[:2]
    max_height = 800
    max_width = 1200
    if height > max_height or width > max_width:
        scale_factor = min(max_height/height, max_width/width)
        input_image = cv2.resize(input_image, None, fx=scale_factor, fy=scale_factor)
        processed_image = cv2.resize(processed_image, None, fx=scale_factor, fy=scale_factor)

    # Step 7: Display the original, processed, and cropped images
    cv2.imshow("Original Image", input_image)
    cv2.imshow("Processed Image", processed_image)
    cv2.imshow("First Letter (ROI)", first_letter_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
