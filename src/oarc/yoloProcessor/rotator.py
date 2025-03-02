import cv2
import numpy as np
import os
import glob

def rotate_in_place(image, angle_deg):
    """
    Rotate the image around its center by angle_deg, preserving the original size.
    Anything that falls outside is cut off, resulting in black corners (or a chosen borderValue).
    """
    (h, w) = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    # Build rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # Warp with the same canvas size => parts outside are discarded
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0)
    )
    return rotated, M

def transform_corners(corners, M):
    """
    corners: (4, 2) array of [x, y] in absolute pixel coords
    M: 2x3 affine transform
    Returns transformed corners of shape (4, 2)
    """
    num_pts = corners.shape[0]  # should be 4
    ones = np.ones((num_pts, 1), dtype=np.float32)
    corners_h = np.hstack([corners, ones])  # (4,3)
    transformed = M.dot(corners_h.T).T      # (4,2)
    return transformed

def main(
    images_folder="path_to_original_images",
    labels_folder="path_to_original_labels",
    output_folder="path_to_augmented_data",
    angle_deg=15.0
):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_folder, base_name + ".txt")
        
        if not os.path.exists(label_path):
            print(f"No label file for {img_path}, skipping.")
            continue
        
        # Read the image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not load {img_path}, skipping.")
            continue
        
        (h, w) = image.shape[:2]
        
        # Rotate image in place
        rotated_img, M = rotate_in_place(image, angle_deg)
        
        # Read corner-based bboxes
        new_bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    # Not in the expected corner format => skip
                    continue
                class_id = int(parts[0])
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[1:])
                
                # Convert normalized corners to absolute pixel coords
                corners = np.array([
                    [x1*w, y1*h],
                    [x2*w, y2*h],
                    [x3*w, y3*h],
                    [x4*w, y4*h],
                ], dtype=np.float32)
                
                # Rotate corners
                corners_rot = transform_corners(corners, M)
                
                # Check if corners are in-bounds => all within [0, w] and [0, h]
                minX, maxX = corners_rot[:,0].min(), corners_rot[:,0].max()
                minY, maxY = corners_rot[:,1].min(), corners_rot[:,1].max()
                
                if (minX < 0 or maxX > w or minY < 0 or maxY > h):
                    # Some (or all) corners are out of the image => discard this box
                    continue
                
                # Otherwise, re-normalize corners
                corners_norm = np.zeros_like(corners_rot)
                corners_norm[:,0] = corners_rot[:,0] / w
                corners_norm[:,1] = corners_rot[:,1] / h
                
                # Flatten in order [x1, y1, x2, y2, x3, y3, x4, y4]
                x1n, y1n = corners_norm[0]
                x2n, y2n = corners_norm[1]
                x3n, y3n = corners_norm[2]
                x4n, y4n = corners_norm[3]
                
                # Store updated corners in the same format
                new_bboxes.append(
                    f"{class_id} {x1n:.6f} {y1n:.6f} {x2n:.6f} {y2n:.6f} {x3n:.6f} {y3n:.6f} {x4n:.6f} {y4n:.6f}"
                )
        
        # Save rotated image
        out_img_path = os.path.join(output_folder, "images", f"{base_name}_rot{int(angle_deg)}.png")
        cv2.imwrite(out_img_path, rotated_img)
        
        # Save new bboxes
        out_label_path = os.path.join(output_folder, "labels", f"{base_name}_rot{int(angle_deg)}.txt")
        with open(out_label_path, 'w') as f_out:
            for bbox_line in new_bboxes:
                f_out.write(bbox_line + "\n")
        
        print(f"Processed {img_path}.  Kept {len(new_bboxes)} bounding boxes after rotation.")


if __name__ == "__main__":
    # Example usage:
    print("Starting data augmentation...")
    for angle in range(-20, 30, 10): 
        main(
            images_folder="m:\\PHOTO_HDD_AUTUMN_GAN\\yoloModelGit\\pcControllerOutput\\images",
            labels_folder="m:\\PHOTO_HDD_AUTUMN_GAN\\yoloModelGit\\pcControllerOutput\\labels",
            output_folder="m:\\PHOTO_HDD_AUTUMN_GAN\\yoloModelGit\\pcControllerOutput\\output",
            angle_deg=angle
        )
    print("Data augmentation complete.")
