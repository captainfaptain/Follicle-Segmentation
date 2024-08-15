import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Load the original image
image_path = r"D:\Project\Follicle_Segmentation\Final Dataset\14.png"
original_image = cv2.imread(image_path)

# Convert to grayscale
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply Fuzzy C-Means Clustering
def fuzzy_c_means(image, clusters=3):
    img_reshape = image.reshape((-1, 1))
    img_reshape = np.float32(img_reshape)

    # Fuzzy C-Means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(img_reshape.T, clusters, 2, error=0.005, maxiter=1000, init=None)
    cluster_membership = np.argmax(u, axis=0)
    segmented_image = cluster_membership.reshape(image.shape)
    
    return segmented_image

segmented_image = fuzzy_c_means(grayscale_image)

# Convert the segmented image to uint8 type for further processing
segmented_image = np.uint8(segmented_image)

# Apply a binary threshold to create a mask of the segmented areas
_, binary_image = cv2.threshold(segmented_image, 1, 255, cv2.THRESH_BINARY)

# Perform Canny edge detection on the binary image
edges = cv2.Canny(binary_image, 50, 150)

# Detect contours on the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for follicles based on contours
follicle_mask = np.zeros_like(grayscale_image)

# Draw only the follicle contours on the mask
for contour in contours:
    area = cv2.contourArea(contour)
    if 100 < area < 5000:  # Adjust these values based on follicle size
        cv2.drawContours(follicle_mask, [contour], -1, 255, thickness=cv2.FILLED)

# Perform edge detection on the follicle mask
follicle_edges = cv2.Canny(follicle_mask, 50, 150)

# Create an image to display detected follicle edges
follicle_detection_image = np.zeros_like(original_image)
follicle_detection_image[follicle_edges > 0] = [0, 0, 255]  # Red color for the follicle edges

# Display the results
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented_image, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Edge Detection")
plt.imshow(edges, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("Follicle Detection")
plt.imshow(cv2.cvtColor(follicle_detection_image, cv2.COLOR_BGR2RGB))

plt.show()
