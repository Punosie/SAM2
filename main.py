import numpy as np
from ultralytics import SAM, YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def segment_with_bboxes(image_path, bbox_list):
    """
    Segments an image using SAM model with bounding box prompts.

    Parameters:
    image_path (str): Path to the input image.
    bbox_list (list[list]): List of bounding boxes where each bounding box is in the format [x_min, y_min, x_max, y_max].

    Returns:
    list: A list of segmented masks if bounding boxes are provided.
    """
    
    model = SAM("model/sam2_t.pt")

    if not bbox_list or len(bbox_list) == 0:
        print("No bounding boxes provided.")
        return None
    
    # Segment with bounding box prompts
    results = model(image_path, bboxes=bbox_list, save=False, show=True)
    
    segmented_masks = results[0].masks.xy

    return segmented_masks


def calculate_centers_with_yolo(image_path):
    """
    Detects objects in the image using YOLOv8n and calculates the centers of the detected bounding boxes.

    Parameters:
    image_path (str): Path to the input image.

    Returns:
    list: A list of centers and bounding boxes for the detected objects in the format [[center_x, center_y, [x_min, y_min, x_max, y_max]]].
    """
    
    # Load the YOLOv8n model
    model = YOLO('model/yolov8n.pt')  # Using YOLOv8n for detection

    # Perform detection on the image
    results = model(image_path)

    centers_and_bboxes = []

    # Loop through each detected object and calculate the center of the bounding box
    for box in results[0].boxes:
        x_min, y_min, x_max, y_max = box.xyxy[0]  # Extract the bounding box coordinates
        
        # Calculate the center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Append the center and bounding box
        centers_and_bboxes.append([center_x, center_y, [x_min, y_min, x_max, y_max]])

    return centers_and_bboxes


def plot_segmented_masks_and_yolo_results(segmented_masks, yolo_results, image_path):
    """
    Plots the SAM-segmented masks, YOLO-detected bounding boxes, and their centers for comparison.

    Parameters:
    segmented_masks (list[numpy.ndarray]): List of segmented masks where each mask is an array of [x, y] coordinates.
    yolo_results (list[list]): YOLO results containing center coordinates and bounding boxes.
    image_path (str): Path to the input image for visualization reference.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the image as a background
    img = plt.imread(image_path)
    ax.imshow(img)

    # Loop through each mask in segmented_masks and plot it
    for i, mask in enumerate(segmented_masks):
        mask = mask.squeeze()  # Remove extra dimensions if needed
        x = mask[:, 0]  # X coordinates
        y = mask[:, 1]  # Y coordinates

        # Compute the centroid (center) of the mask
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)

        # Plot the mask
        ax.plot(x, y, marker='o', markersize=2, linestyle='-', label=f'SAM Mask {i+1}')

        # Plot the centroid (center) of the SAM-segmented mask
        ax.plot(centroid_x, centroid_y, 'rx', markersize=10, label=f'SAM Center {i+1}')

    # Loop through YOLO-detected results and plot the bounding boxes and centers
    for j, (center_x, center_y, bbox) in enumerate(yolo_results):
        x_min, y_min, x_max, y_max = bbox

        # Plot the bounding box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='g', facecolor='none', label=f'YOLO BBox {j+1}')
        ax.add_patch(rect)

        # Plot the center of the YOLO-detected bounding box
        ax.plot(center_x, center_y, 'bo', markersize=10, label=f'YOLO Center {j+1}')

        # Annotate the YOLO center
        ax.text(center_x, center_y, f'({center_x:.2f}, {center_y:.2f})', fontsize=12, color='blue')

    plt.title("Segmented Masks, YOLO Bounding Boxes, and Their Centers")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.legend()
    plt.show()


if __name__ == "__main__":
    image_path = "assets/person.jpg"
    bbox_list = [[100, 50, 600, 400]]
    
    # Segment the image using SAM
    segmented_masks = segment_with_bboxes(image_path, bbox_list)
    
    # Calculate YOLO centers and bounding boxes
    yolo_results = calculate_centers_with_yolo(image_path)
    
    # Plot the segmented masks and YOLO results for comparison
    if segmented_masks and yolo_results:
        plot_segmented_masks_and_yolo_results(segmented_masks, yolo_results, image_path)
