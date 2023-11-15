import cv2
import numpy as np
import os

# Path to the directory containing the images
image_dir = "train"  # Replace with your actual path

# Loop through the images in the directory
for filename in os.listdir(image_dir):
    # Construct the full path to the image file
    image_path = os.path.join(image_dir, filename)

    # Load the image
    image = cv2.imread('test2.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Perform a dilation operation to close gaps in between object edges
    dilated_edges = cv2.dilate(edges, None, iterations=2)

    # Find contours of the objects from the dilated edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to keep only those that represent the road
    road_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Adjust the area threshold as needed
            road_contours.append(contour)

    # Create a blank image with the same size as the grayscale image for the sketch map
    sketch_map = np.ones_like(gray) * 255

    # Draw the road boundaries on the image and sketch map
    cv2.drawContours(image, road_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(sketch_map, road_contours, -1, (0, 0, 0), 2)

    # Display the image with road boundaries
    cv2.imshow('Road Boundaries', image)

    # Display the sketch map with road boundaries
    cv2.imshow('Sketch Map with Road Boundaries', sketch_map)

    cv2.waitKey(0)

cv2.destroyAllWindows()