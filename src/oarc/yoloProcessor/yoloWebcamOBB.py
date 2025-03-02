import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import pyvirtualcam
from pyvirtualcam import PixelFormat

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class yoloWebcam:
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.model = None
        self.color_map = {}
        # Define neon colors in BGR format
        self.neon_colors = [
            (0, 255, 255),    # Neon Yellow
            (255, 0, 255),    # Neon Pink
            (0, 255, 0),      # Neon Green
            (255, 128, 0),    # Neon Blue
            (0, 128, 255),    # Neon Orange
            (255, 0, 128),    # Neon Purple
            (0, 255, 128),    # Neon Turquoise
            (128, 255, 0),    # Neon Lime
        ]
        self.color_index = 0
        self.load_model()

    def load_model(self):
        """Load YOLO model from environment variable path"""
        model_dir = os.getenv('YOLO_MODEL_GIT')
        if model_dir is None:
            raise EnvironmentError("YOLO_MODEL_GIT environment variable not set")

        model_name = "my_model.pt"  # Update this to your OBB model name
        model_path = os.path.join(model_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            print(f"Loaded model from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def get_color_for_class(self, class_name):
        """Assign a unique neon color for each class"""
        if class_name not in self.color_map:
            # Assign next available neon color
            color = self.neon_colors[self.color_index % len(self.neon_colors)]
            self.color_map[class_name] = color
            self.color_index += 1
        return self.color_map[class_name]

    def draw_oriented_bbox(self, img, points, color, label=None):
        """Draw oriented bounding box using polygon points"""
        points = points.astype(np.int32)
        
        # Draw the oriented bounding box
        cv2.polylines(img, [points], True, color, 2)
        
        # Add label if provided
        if label:
            # Get top-left corner for label placement
            x_min = min(points[:, 0])
            y_min = min(points[:, 1])
            cv2.putText(img, label, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def process_frame(self, frame):
        """Process frame with YOLO OBB detection"""
        if self.model is None:
            return frame

        # Run inference
        results = self.model(frame, verbose=False)
        
        # Draw detections
        frame_out = frame.copy()
        for r in results:
            # Check if oriented bounding boxes exist in results
            if hasattr(r, 'obb') and r.obb is not None:
                boxes = r.obb.xyxyxyxy  # Get polygon coordinates
                confs = r.obb.conf      # Get confidence scores
                classes = r.obb.cls     # Get class indices
                
                # Process each detection
                for box, conf, cls_idx in zip(boxes, confs, classes):
                    if conf >= self.conf_threshold:
                        # Convert box to numpy array and reshape to polygon format
                        points = box.cpu().numpy().reshape((-1, 2))
                        
                        # Get class name and color
                        cls_idx = int(cls_idx.item())
                        name = self.model.names[cls_idx]
                        color = self.get_color_for_class(name)
                        
                        # Draw oriented bounding box with label
                        label = f'{name}: {conf:.2f}'
                        self.draw_oriented_bbox(frame_out, points, color, label)
        
        return frame_out

    def virtual_webcam_stream(self):
        """Stream YOLO detection to virtual webcam"""
        cap = cv2.VideoCapture(0)
        
        # Get webcam resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        with pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=PixelFormat.BGR, backend='obs') as cam:
            print(f'Using virtual camera: {cam.device}')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame with YOLO detection
                processed_frame = self.process_frame(frame)
                
                # Send to virtual camera
                cam.send(processed_frame)
                cam.sleep_until_next_frame()
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        cap.release()

if __name__ == "__main__":
    yolo_cam = yoloWebcam(conf_threshold=0.5)
    yolo_cam.virtual_webcam_stream()