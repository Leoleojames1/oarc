import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import win32gui
import win32ui
import win32con
import win32api
import os
import json
import websockets
from typing import Optional, Dict, Any, Tuple

class yoloProcessor:
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):
        """
        Initialize yoloProcessor with configurable model path and confidence threshold
        
        Args:
            model_path (str): Path to YOLO model file (.pt)
            conf_threshold (float): Confidence threshold for detections (0-1)
        """
        self.model = None
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Load a YOLO model from specified path
        
        Args:
            model_path (str): Path to YOLO model file
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            self.model = YOLO(model_path)
            self.model_path = model_path
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def capture_screen(self) -> np.ndarray:
        """Capture screen content and return as numpy array"""
        hwin = win32gui.GetDesktopWindow()
        
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        
        memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)
        
        # Clean up resources
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    def detect_objects(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, list]:
        """
        Detect objects in frame using loaded YOLO model
        
        Args:
            frame (np.ndarray): Input image/frame
            draw (bool): Whether to draw detection boxes
            
        Returns:
            Tuple[np.ndarray, list]: Processed frame and list of detections
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Run inference
        results = self.model(frame, verbose=False)
        
        # Process detections
        detections = []
        frame_out = frame.copy() if draw else frame
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model.names[cls]
                
                if conf >= self.conf_threshold:
                    detection = {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class": cls,
                        "name": name
                    }
                    detections.append(detection)
                    
                    if draw:
                        # Draw detection
                        cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_out, f'{name}: {conf:.2f}', 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)
        
        return frame_out, detections

    def process_screen(self) -> Tuple[np.ndarray, list]:
        """Process current screen content with object detection"""
        try:
            frame = self.capture_screen()
            return self.detect_objects(frame)
        except Exception as e:
            print(f"Error processing screen: {e}")
            return np.zeros((100, 100, 3), dtype=np.uint8), []

    def create_gradio_interface(self, title: str = "YOLO Object Detector"):
        """Create Gradio interface for the detector"""
        demo = gr.Interface(
            fn=self.process_screen,
            inputs=None,
            outputs="image",
            live=True,
            title=title,
            description=f"Real-time object detection using {os.path.basename(self.model_path)}",
            theme=gr.themes.Base(),
        )
        return demo
    
    async def send_yolo_response_to_frontend(self, response: Dict[str, Any]):
        """Send YOLO detection response to frontend"""
        async with websockets.connect('ws://localhost:2020/yolo_stream') as websocket:
            await websocket.send(json.dumps(response))
            
if __name__ == "__main__":
    # Example usage
    model_name_string = r"my_model"
    model_name = rf"{model_name_string}.pt"
    model_git_dir = os.getenv('OARC_MODEL_GIT')
    
    # Load model from git directory if available
    if model_git_dir:
        model_path = os.path.join(model_git_dir, model_name)
    
    detector = yoloProcessor(model_path, conf_threshold=0.5)
    demo = detector.create_gradio_interface()
    demo.launch(share=False)