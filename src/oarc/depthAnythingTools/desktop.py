import gradio as gr
import cv2
import numpy as np
import torch
import sys
import os
import pyvirtualcam
from pyvirtualcam import PixelFormat
from huggingface_hub import hf_hub_download
from mss import mss

sys.path.append('M:/PHOTO_HDD_AUTUMN_GAN/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# Device selection with MPS support
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Model configurations
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder2name = {
    'vits': 'Small',
    'vitb': 'Base',
    'vitl': 'Large'
}

# Model IDs and filenames for HuggingFace Hub
MODEL_INFO = {
    'vits': {
        'repo_id': 'depth-anything/Depth-Anything-V2-Small',
        'filename': 'depth_anything_v2_vits.pth'
    },
    'vitb': {
        'repo_id': 'depth-anything/Depth-Anything-V2-Base',
        'filename': 'depth_anything_v2_vitb.pth'
    },
    'vitl': {
        'repo_id': 'depth-anything/Depth-Anything-V2-Large',
        'filename': 'depth_anything_v2_vitl.pth'
    }
}

# Global variables for model management
current_model = None
current_encoder = None

def download_model(encoder):
    """Download the specified model from HuggingFace Hub"""
    model_info = MODEL_INFO[encoder]
    model_path = hf_hub_download(
        repo_id=model_info['repo_id'],
        filename=model_info['filename'],
        local_dir='checkpoints'
    )
    return model_path

def load_model(encoder):
    """Load the specified model"""
    global current_model, current_encoder
    if current_encoder != encoder:
        model_path = download_model(encoder)
        current_model = DepthAnythingV2(**model_configs[encoder])
        current_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        current_model = current_model.to(DEVICE).eval()
        current_encoder = encoder
    return current_model

@torch.inference_mode()
def predict_depth(image, encoder):
    """Predict depth using the selected model"""
    model = load_model(encoder)
    depth = model.infer_image(image)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return depth

def process_frame(frame, encoder):
    """Process a single frame from the desktop capture"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth = predict_depth(frame_rgb, encoder)
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    return depth_colored

def desktop_depth_stream(encoder):
    """Stream depth display of desktop capture to virtual webcam"""
    with mss() as sct:
        monitor = sct.monitors[1]  # Capture the primary monitor
        with pyvirtualcam.Camera(width=monitor["width"], height=monitor["height"], fps=30, fmt=PixelFormat.BGR) as cam:
            print(f'Using virtual camera: {cam.device}')
            while True:
                frame = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                depth_colored = process_frame(frame, encoder)
                cam.send(depth_colored)
                cam.sleep_until_next_frame()

def start_virtual_webcam(model_name):
    encoder = {v: k for k, v in encoder2name.items()}[model_name]
    desktop_depth_stream(encoder)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Depth Anything V2 Virtual Webcam for Desktop Capture")
    
    model_dropdown = gr.Dropdown(
        choices=list(encoder2name.values()),
        value="Small",
        label="Select Model Size"
    )
    
    start_button = gr.Button("Start Virtual Webcam")
    
    start_button.click(fn=start_virtual_webcam, inputs=model_dropdown, outputs=[])

if __name__ == "__main__":
    demo.launch()