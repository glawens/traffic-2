import os
import cv2
import numpy as np

def resize_image(image, target_size=(300, 300)):
    """
    Resize the image to the target size while maintaining the aspect ratio.
    """
    height, width = image.shape[:2]
    if width > height:
        scale = target_size[0] / width
    else:
        scale = target_size[1] / height
    resized_image = cv2.resize(image, None, fx=scale, fy=scale)
    return resized_image

def remove_background(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert the binary image
    thresh = cv2.bitwise_not(thresh)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the image
    mask = np.zeros_like(gray)

    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Bitwise AND operation to remove the background
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

def add_white_background(image, target_size=(300, 300)):
    # Create a white background
    white_bg = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255

    # Calculate offset to center the image on the white background
    offset_x = (target_size[0] - image.shape[1]) // 2
    offset_y = (target_size[1] - image.shape[0]) // 2

    # Paste the image onto the white background
    white_bg[offset_y:offset_y+image.shape[0], offset_x:offset_x+image.shape[1]] = image

    return white_bg



# Path to the folder containing images
folder_path = 'C:\\Users\\Eric\\Desktop\\train1\\speed_80'

# Output folder to save processed images
output_folder = 'C:\\Users\\Eric\\Desktop\\train1\\speed_80\\speed_80'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Process only image files
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = resize_image(image)

        # Remove background
        image_without_bg = remove_background(resized_image)

        # Add white background
        image_with_white_bg = add_white_background(image_without_bg)

        # Save the processed image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image_with_white_bg)

        print(f"Processed: {filename}")

print("All images processed successfully!")
