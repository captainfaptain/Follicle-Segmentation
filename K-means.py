import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Function to calculate texture features
def calculate_texture_features(image):
    # Compute GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

    # Compute texture properties from GLCM
    props = ['energy', 'contrast', 'homogeneity', 'dissimilarity', 'correlation']
    texture_features = [graycoprops(glcm, prop).ravel()[0] for prop in props]

    # Compute standard deviation
    std_deviation = np.std(image)

    return texture_features, std_deviation

# Function to process a single image
def process_image(image_path):
    # Load the ultrasound image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply K-means clustering
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(gray.reshape(-1, 1))
        inertia.append(kmeans.inertia_)

    # Find the number of clusters where the curve starts
    for i in range(1, len(inertia)):
        if inertia[i] - inertia[i-1] < 0:
            num_clusters = i + 1
            break

    # Apply K-means clustering with the determined number of clusters
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(gray.reshape(-1, 1))
    segmented_labels = kmeans.labels_.reshape(gray.shape)

    # Make the segmented image grayscale
    segmented_img = np.zeros_like(image)
    for i in range(num_clusters):
        segmented_img[segmented_labels == i] = np.mean(image[segmented_labels == i].astype(float), axis=0)
    segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

    # Thresholding to binarize the image
    _, thresh = cv2.threshold(segmented_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determine bounding box of the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Adjust the bounding box to include some margin
    margin = -23
    x = max(0, x - margin)
    y = max(0, y - margin)
    w += 2 * margin
    h += 2 * margin

    # Crop the region of interest from the segmented image
    cropped_segmented_img = segmented_img[y:y+h, x:x+w]

    # Preprocessing
    cropped_equalized = cv2.equalizeHist(cropped_segmented_img)
    bilateral_filtered = cv2.bilateralFilter(cropped_equalized, 9, 75, 75)
    median_filtered = cv2.medianBlur(bilateral_filtered, 5)

    _, thresh = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank canvas with black background
    follicle_image = np.zeros_like(cropped_segmented_img)

    # Draw filled contours of follicles on the blank canvas
    cv2.drawContours(follicle_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Preprocess the segmented image
    binary_mask = np.uint8(follicle_image) # type: ignore

    # Apply morphological operations (e.g., dilation and erosion) for refining the segmentation
    dilation = cv2.dilate(binary_mask, kernel, iterations=1) # type: ignore

    # Apply Canny edge detection for better edge visualization
    edges = cv2.Canny(cropped_segmented_img, 100, 200)

    # Label connected components (follicle regions)
    labeled_image = measure.label(dilation, connectivity=2)
    regions = measure.regionprops(labeled_image)

    # Extract relevant properties or features of follicle regions
    follicle_areas = []
    for region in regions:
        if region.area > 100:  # Filtering small regions
            follicle_areas.append(region)

    # Compute texture features and standard deviation
    texture_features, std_deviation = calculate_texture_features(cropped_segmented_img)
    energy, contrast, homogeneity, dissimilarity, _ = texture_features

    # Compute clustering metrics
    segmented_labels_flat = segmented_labels.flatten()
    silhouette = silhouette_score(gray.reshape(-1, 1), segmented_labels_flat)
    davies_bouldin = davies_bouldin_score(gray.reshape(-1, 1), segmented_labels_flat)

    # Return relevant variables along with other variables
    return image, segmented_img, cropped_segmented_img, follicle_image, num_clusters, follicle_areas, x, y, edges, energy, contrast, homogeneity, dissimilarity, std_deviation, silhouette, davies_bouldin

# Specify the folder containing the images
folder_path = r"D:\Project\Follicle_Segmentation\Final Dataset"

# Get a list of all files in the folder
image_files = os.listdir(folder_path)

# Iterate through each image file
for filename in image_files:
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Filter only jpg or png files
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image, segmented_img, cropped_segmented_img, follicle_image, num_clusters, follicle_areas, x, y, edges, energy, contrast, homogeneity, dissimilarity, std_deviation, silhouette, davies_bouldin = process_image(image_path)

        # Display the plots
        plt.figure(figsize=(18, 12), facecolor='white')

        # Original Image
        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image', color='black')
        plt.axis('off')

        # Segmented Image
        plt.subplot(2, 3, 2)
        plt.imshow(segmented_img, cmap='gray')
        plt.title('Segmented Image', color='black')
        plt.axis('off')

        # Texture Features
        plt.subplot(2, 3, 3)
        plt.text(0.1, 0.9, r"$\bf{File\ Name:}$ " + f"{filename}\n\n" +
                         r"$\bf{Texture\ Features:}$" + "\n" +
                         f"Energy: {energy:.2f}\n" +
                         f"Contrast: {contrast:.2f}\n" +
                         f"Homogeneity: {homogeneity:.2f}\n" +
                         f"Dissimilarity: {dissimilarity:.2f}\n" +
                         f"Standard Deviation: {std_deviation:.2f}\n\n" +
                         r"$\bf{Clustering\ Metrics:}$" + "\n" +
                         f"Silhouette Score: {silhouette:.4f}\n" +
                         f"Davies-Bouldin Index: {davies_bouldin:.4f}",
                 horizontalalignment='left',
                 verticalalignment='top',
                 fontsize=14,
                 color='black')
        plt.axis('off')
        plt.axhline(y=0.85, color='black', linestyle='-', linewidth=1)

        # Follicle Detection
        plt.subplot(2, 3, 4)
        plt.imshow(cropped_segmented_img, cmap='gray')
        plt.contour(follicle_image, colors='red', linewidths=2)
        plt.title(f'Follicle Detection (Number={len(follicle_areas)})', color='black')
        plt.axis('off')

        # Edge Detection
        plt.subplot(2, 3, 5)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection', color='black')
        plt.axis('off')

        # Add horizontal line separator
        plt.axhline(y=0.5, color='black', linestyle='-', linewidth=2)

        plt.tight_layout()
        plt.show()
