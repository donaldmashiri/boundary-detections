import cv2
import numpy as np
import os
import json
from geojson import Feature, Polygon, FeatureCollection

image_path = "test2.jpg"
output_dir = "output"

print("testing")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the image
image = cv2.imread(image_path)
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

# Save the sketch map
sketch_map_path = os.path.join(output_dir, "sketch_map.jpg")
cv2.imwrite(sketch_map_path, sketch_map)

# Save the original image with road boundaries
image_with_boundaries_path = os.path.join(output_dir, "image_with_boundaries.jpg")
cv2.imwrite(image_with_boundaries_path, image)

cv2.imshow('Road Boundaries', image)
cv2.imshow('Sketch Map with Road Boundaries', sketch_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a GeoJSON feature collection
features = []
coordinates = []

for contour in road_contours:
    contour = contour.reshape(-1, 2).tolist()
    coordinates.append(contour)

# Create a GeoJSON polygon feature
if coordinates:
    polygon = Polygon(coordinates)
    feature = Feature(geometry=polygon)
    features.append(feature)

# Create a feature collection
feature_collection = FeatureCollection(features)

# Save the GeoJSON file
geojson_path = os.path.join(output_dir, "results.geojson")
with open(geojson_path, 'w') as f:
    json.dump(feature_collection, f)

print("Sketch map, image with road boundaries, and results.geojson saved successfully.")