import cv2
import numpy as np
import os
import json
from geojson import Feature, Polygon, FeatureCollection

image_dir = "train"

# model images trained for boundary detection
# import yaml
# with open('data.yaml', 'r') as file:
#     data = yaml.safe_load(file)

for filename in os.listdir(image_dir):

    image_path = os.path.join(image_dir, filename)

    # Load the image
    image = cv2.imread('test.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)


    dilated_edges = cv2.dilate(edges, None, iterations=2)

    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    road_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            road_contours.append(contour)

    sketch_map = np.ones_like(gray) * 255

    cv2.drawContours(image, road_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(sketch_map, road_contours, -1, (0, 0, 0), 2)

    cv2.imshow('Road Boundaries', image)

    cv2.imshow('Sketch Map with Road Boundaries', sketch_map)

    cv2.waitKey(0)

cv2.destroyAllWindows()


