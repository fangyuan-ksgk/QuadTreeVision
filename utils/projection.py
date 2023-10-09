import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import numpy as np

# Pytorch Implementation
def compute_corners_3d_local(center_3d, dimensions_3d):
    l, w, h = dimensions_3d
    
    # Define half lengths, widths, and heights
    half_l = l / 2
    half_w = w / 2
    half_h = h / 2
    
    # Define corner offsets
    corner_offsets = [
        torch.tensor([half_l, half_w, half_h]),
        torch.tensor([half_l, -half_w, half_h]),
        torch.tensor([-half_l, half_w, half_h]),
        torch.tensor([-half_l, -half_w, half_h]),
        torch.tensor([half_l, half_w, -half_h]),
        torch.tensor([half_l, -half_w, -half_h]),
        torch.tensor([-half_l, half_w, -half_h]),
        torch.tensor([-half_l, -half_w, -half_h])
    ]
    
    # Calculate corner positions in local coordinates
    corners_3d_local = []
    for offset in corner_offsets:
        corner = center_3d + offset
        corners_3d_local.append(corner)
    
    return torch.stack(corners_3d_local)

def project_to_image_plane(corners_3d_camera, intrinsic_matrix):
    # Apply perspective projection to 3D corners
    # Assumes corners_3d_camera is of shape (8, 3) and intrinsic_matrix is a 3x3 matrix
    
    # Apply perspective projection
    corners_2d_pixel = torch.matmul(intrinsic_matrix, corners_3d_camera.T)
    
    # Normalize by the last row (z-coordinate)
    corners_2d_pixel = corners_2d_pixel / corners_2d_pixel[2]
    
    return corners_2d_pixel[:2].T


def visualize_2d_corners_with_edges(image, corners_2d_pixel, dimensions, location, text='3D Voxel'):
    # Load the image
    # image = plt.imread(image_path)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(image)
    
    # Plot the projected 2D corners as red circles
    ax.plot(corners_2d_pixel[:, 0], corners_2d_pixel[:, 1], 'ro', markersize=3)
    
    # Plot edges of the 3D rectangle
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        ax.plot(corners_2d_pixel[edge, 0], corners_2d_pixel[edge, 1], 'r--', linewidth=1.0, alpha=0.5)
        
    # Expanded 2DBox on image
    expanded_2d_box = get_expanded_2d_box(corners_2d_pixel)
    
    # Create a rectangle patch for the expanded 2D bounding box
    rect = Rectangle((expanded_2d_box[0], expanded_2d_box[1]), expanded_2d_box[2], expanded_2d_box[3],
                     linewidth=1, edgecolor='r', facecolor='y', alpha=0.5)
    
    # Add the rectangle patch to the axis
    ax.add_patch(rect)
    
    text_x = expanded_2d_box[0] + expanded_2d_box[2] / 2
    text_y = expanded_2d_box[1] - 2  # Adjust this value for text position
    ax.text(text_x, text_y, text, fontsize=6, color='r', ha='center')
    
    # Set plot limits to match image dimensions
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    
    # Show the plot
    plt.show()
    
    
def get_expanded_2d_box(corners_2d_pixel):
    # Calculate the bounding box coordinates
    min_x = torch.min(corners_2d_pixel[:, 0])
    max_x = torch.max(corners_2d_pixel[:, 0])
    min_y = torch.min(corners_2d_pixel[:, 1])
    max_y = torch.max(corners_2d_pixel[:, 1])
    
    # Create the expanded 2D box as [x_min, y_min, width, height]
    expanded_2d_box = torch.stack([min_x, min_y, max_x - min_x, max_y - min_y])
    
    return expanded_2d_box


def compute_corners_3d_local(center_3d, dimensions_3d):
    l, w, h = dimensions_3d
    # yaw, pitch, roll = orientation_3d
    # Calculate the rotation matrix using yaw, pitch, and roll
    # rotation_matrix = get_rotation_matrix(yaw, pitch, roll)
    
    # Define half lengths, widths, and heights
    half_l = l / 2
    half_w = w / 2
    half_h = h / 2
    
    # Define corner offsets
    corner_offsets = [
        torch.tensor([half_l, half_w, half_h]),
        torch.tensor([half_l, -half_w, half_h]),
        torch.tensor([-half_l, half_w, half_h]),
        torch.tensor([-half_l, -half_w, half_h]),
        torch.tensor([half_l, half_w, -half_h]),
        torch.tensor([half_l, -half_w, -half_h]),
        torch.tensor([-half_l, half_w, -half_h]),
        torch.tensor([-half_l, -half_w, -half_h])
    ]
    
    # Calculate corner positions in local coordinates
    corners_3d_local = []
    for offset in corner_offsets:
        # rotated_offset = torch.matmul(rotation_matrix, offset)
        corner = center_3d + offset
        corners_3d_local.append(corner)
    
    return torch.stack(corners_3d_local)

def expand_and_project_bbox(center_3d, dimensions_3d, intrinsic_matrix):
    # Define the corners of the 3D bounding box in local coordinate space
    corners_3d_camera = compute_corners_3d_local(center_3d, dimensions_3d)
    # Project 3D camera onto 2D image plane
    corners_2d_pixel = project_to_image_plane(corners_3d_camera, intrinsic_matrix)
    # Find the minimum and maximum coordinates for x and y axes
    min_x, _ = torch.min(corners_2d_pixel[:, 0], dim=0)
    max_x, _ = torch.max(corners_2d_pixel[:, 0], dim=0)
    min_y, _ = torch.min(corners_2d_pixel[:, 1], dim=0)
    max_y, _ = torch.max(corners_2d_pixel[:, 1], dim=0)
    
    return min_x, min_y, max_x-min_x, max_y-min_y


def draw_translucent_rectangle(image, rect_coords, color, alpha):
    overlay = image.copy()
    cv2.rectangle(overlay, rect_coords[0], rect_coords[1], color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
def visualize_2d_corners_with_edges_cv(orig_image, corners_2d_pixel, dimensions, location, text='3D Voxel'):
    # Create a copy of the image to draw on
    image = orig_image.copy()
    # image_with_drawings = image.copy()
    # Expanded 2DBox on image
    expanded_2d_box = get_expanded_2d_box(corners_2d_pixel)
    # Convert PyTorch tensor to NumPy array
    corners_2d_pixel = corners_2d_pixel.cpu().numpy()
    # Draw the projected 2D corners as red circles
    for corner in corners_2d_pixel:
        cv2.circle(image, tuple(corner.astype(int)), 1, (255, 0, 0), -1)
    
    # Draw edges of the 3D rectangle
    edges = [
        [0, 1], [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        cv2.line(image, tuple(corners_2d_pixel[edge[0]].astype(int)),
                 tuple(corners_2d_pixel[edge[1]].astype(int)), (255, 0, 0), 1, cv2.LINE_AA)
    
    
    expanded_box_color = (100, 100, 0)  # Yellow color
    
    rect_coords = (int(expanded_2d_box[0]), int(expanded_2d_box[1])), (int(expanded_2d_box[0] + expanded_2d_box[2]), int(expanded_2d_box[1] + expanded_2d_box[3]))
    draw_translucent_rectangle(image, rect_coords, expanded_box_color, alpha=0.005)
    # Draw the expanded 2D bounding box
    # cv2.rectangle(image_with_drawings, (int(expanded_2d_box[0]), int(expanded_2d_box[1])),
    #               (int(expanded_2d_box[0] + expanded_2d_box[2]), int(expanded_2d_box[1] + expanded_2d_box[3])),
    #               expanded_box_color, 1)
    
    # Calculate text position
    text_x = int(expanded_2d_box[0])
    text_y = int(expanded_2d_box[1] - 2)
    
    # Draw text
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    return image