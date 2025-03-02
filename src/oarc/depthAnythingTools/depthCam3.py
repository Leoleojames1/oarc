import gradio as gr
import cv2
import numpy as np
import torch
import sys
import os

# Add the repository root to Python path
sys.path.append('M:/PHOTO_HDD_AUTUMN_GAN/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# Device selection with MPS support
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Performance settings
PROCESS_WIDTH = 384  # Width to resize to for processing
PROCESS_HEIGHT = 288  # Height to resize to for processing

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder2name = {
    'vits': 'Small',
    'vitb': 'Base',
    'vitl': 'Large',
    'vitg': 'Giant'
}

# Global variables for model management
current_model = None
current_encoder = None

def load_model(encoder):
    """Load the specified model"""
    global current_model, current_encoder
    if current_encoder != encoder:
        try:
            current_model = DepthAnythingV2(**model_configs[encoder])
            weights_path = f'M:/PHOTO_HDD_AUTUMN_GAN/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights not found at {weights_path}.")
            current_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            current_model = current_model.to(DEVICE).eval()
            # Enable CUDNN benchmarking for faster performance
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
            current_encoder = encoder
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    return current_model

@torch.inference_mode()  # Faster than with torch.no_grad()
def predict_depth(image, encoder):
    """Predict depth using the selected model"""
    model = load_model(encoder)
    depth = model.infer_image(image)
    # Normalize depth map
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return depth

def process_frame(frame, encoder):
    """Process a single frame from the webcam"""
    try:
        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        depth = predict_depth(frame_resized, encoder)
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        
        # Resize back to original size
        depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
        return depth_colored
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

def webcam_stream(model_name):
    """Stream webcam with depth visualization"""
    encoder = {v: k for k, v in encoder2name.items()}[model_name]
    
    cap = cv2.VideoCapture(0)
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        yield None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            depth_colored = process_frame(frame, encoder)
            if depth_colored is not None:
                yield depth_colored
            
    finally:
        cap.release()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Depth Anything V2 Webcam Depth Stream")
    
    model_dropdown = gr.Dropdown(
        choices=list(encoder2name.values()),
        value="Small",
        label="Select Model Size"
    )
    
    output = gr.Image(label="Depth Map", streaming=True)
    
    start_btn = gr.Button("Start Stream")
    
    start_btn.click(
        fn=webcam_stream,
        inputs=[model_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    demo.queue().launch()