import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import pyvirtualcam
from pyvirtualcam import PixelFormat
from time import time

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class DetectedObject:
    def __init__(self, points, class_name, confidence):
        self.points = points
        self.class_name = class_name
        self.confidence = confidence
        self.last_seen = time()
        self.is_visible = True

    def update(self, points, confidence):
        self.points = points
        self.confidence = confidence
        self.last_seen = time()
        self.is_visible = True

    def check_visibility(self, debounce_time):
        if self.is_visible and time() - self.last_seen > debounce_time:
            self.is_visible = False
        return self.is_visible

class yoloWebcam:
    def __init__(self, conf_threshold=0.5, debounce_ms=200):
        self.conf_threshold = conf_threshold
        self.debounce_time = debounce_ms / 1000.0  # Convert to seconds
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
        self.tracked_objects = []
        self.load_model()

    def load_model(self):
        """Load YOLO model from environment variable path"""
        model_dir = os.getenv('YOLO_MODEL_GIT')
        if model_dir is None:
            raise EnvironmentError("YOLO_MODEL_GIT environment variable not set")

        model_name = "pcController.pt"  # Update this to your OBB model name
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
            color = self.neon_colors[self.color_index % len(self.neon_colors)]
            self.color_map[class_name] = color
            self.color_index += 1
        return self.color_map[class_name]

    def find_matching_object(self, points, class_name):
        """Find the closest matching object of the same class"""
        min_distance = float('inf')
        matching_object = None
        
        center = np.mean(points, axis=0)
        
        for obj in self.tracked_objects:
            if obj.class_name == class_name:
                obj_center = np.mean(obj.points, axis=0)
                distance = np.linalg.norm(center - obj_center)
                
                if distance < min_distance:
                    min_distance = distance
                    matching_object = obj
                    
        # Only match if the centers are reasonably close
        if min_distance > 50:  # Adjust this threshold as needed
            matching_object = None
            
        return matching_object

    def draw_oriented_bbox(self, img, points, color, label=None):
        """Draw oriented bounding box using polygon points"""
        points = points.astype(np.int32)
        
        # Draw the oriented bounding box
        cv2.polylines(img, [points], True, color, 2)
        
        # Add label if provided
        if label:
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
        
        # Track current detections
        current_detections = set()
        
        # Draw detections
        frame_out = frame.copy()
        for r in results:
            if hasattr(r, 'obb') and r.obb is not None:
                boxes = r.obb.xyxyxyxy
                confs = r.obb.conf
                classes = r.obb.cls
                
                for box, conf, cls_idx in zip(boxes, confs, classes):
                    if conf >= self.conf_threshold:
                        points = box.cpu().numpy().reshape((-1, 2))
                        cls_idx = int(cls_idx.item())
                        name = self.model.names[cls_idx]
                        
                        # Find or create tracked object
                        tracked_obj = self.find_matching_object(points, name)
                        if tracked_obj is None:
                            tracked_obj = DetectedObject(points, name, conf)
                            self.tracked_objects.append(tracked_obj)
                        else:
                            tracked_obj.update(points, conf)
                            
                        current_detections.add(tracked_obj)
        
        # Update visibility of objects not detected in current frame
        for obj in self.tracked_objects:
            if obj not in current_detections:
                obj.check_visibility(self.debounce_time)
        
        # Draw only visible objects
        for obj in self.tracked_objects:
            if obj.is_visible:
                color = self.get_color_for_class(obj.class_name)
                label = f'{obj.class_name}: {obj.confidence:.2f}'
                self.draw_oriented_bbox(frame_out, obj.points, color, label)
        
        # Clean up objects that haven't been seen for a while
        self.tracked_objects = [obj for obj in self.tracked_objects 
                              if time() - obj.last_seen < self.debounce_time * 5]
        
        return frame_out

    def virtual_webcam_stream(self):
        """Stream YOLO detection to virtual webcam"""
        cap = cv2.VideoCapture(0)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        with pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=PixelFormat.BGR, backend='obs') as cam:
            print(f'Using virtual camera: {cam.device}')
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                
                cam.send(processed_frame)
                cam.sleep_until_next_frame()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        cap.release()

if __name__ == "__main__":
    # Initialize with 200ms debounce time and 0.5 confidence threshold
    # yolo_cam = yoloWebcam(conf_threshold=0.5, debounce_ms=200)
    # More strict confidence, longer debounce
    # yolo_cam = yoloWebcam(conf_threshold=0.7, debounce_ms=500)
    yolo_cam = yoloWebcam(conf_threshold=0.7, debounce_ms=20)
    yolo_cam.virtual_webcam_stream()