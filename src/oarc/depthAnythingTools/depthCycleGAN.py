import gradio as gr
import cv2
import numpy as np
import torch
import os
import sys
import shutil
import tempfile
import pandas as pd
from PIL import Image
from huggingface_hub import HfApi, HfFolder, hf_hub_download, create_repo
from datasets import Dataset, Features, Image as DatasetsImage
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import time

# Add the repository root to Python path
sys.path.append('M:/PHOTO_HDD_AUTUMN_GAN/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Model configurations (assuming these are defined in your code)
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

name2encoder = {v: k for k, v in encoder2name.items()}

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
    if model is None:
        raise ValueError(f"Model for encoder {encoder} could not be loaded.")
    if isinstance(image, Image.Image):
        image = np.array(image)
    depth = model.infer_image(image)
    # Normalize depth map
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    return depth

def resize_image(image, max_size=1200):
    """Resize image if its dimensions exceed max_size"""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    return image

def save_image(image, path):
    """Save PIL Image to the specified path"""
    image.save(path, format="PNG")

def process_images(folder_path, encoder):
    """Process all images in the folder and generate depth maps"""
    images = []
    depth_maps = []
    temp_dir = tempfile.mkdtemp()
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            temp_image_path = os.path.join(temp_dir, filename)
            shutil.copy(image_path, temp_image_path)
            image = Image.open(temp_image_path).convert('RGB')
            image = resize_image(image)
            image_np = np.array(image)
            depth_map = predict_depth(image_np, encoder)
            depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            save_image(image, temp_image_path)
            depth_map_path = os.path.join(temp_dir, f"depth_{filename}")
            save_image(Image.fromarray(depth_map_colored), depth_map_path)
            images.append(temp_image_path)
            depth_maps.append(depth_map_path)
            print(f"Processed {filename}")
    return images, depth_maps, temp_dir

def upload_to_hf(images, depth_maps, repo_id):
    """Upload images and depth maps to Hugging Face Hub"""
    api = HfApi()
    token = HfFolder.get_token()
    
    # Ensure the repository exists and is of type 'dataset'
    try:
        api.repo_info(repo_id=repo_id, token=token)
    except Exception as e:
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", token=token)
        except Exception as create_e:
            if "You already created this dataset repo" not in str(create_e):
                raise create_e
    
    for image_path, depth_map_path in zip(images, depth_maps):
        api.upload_file(
            path_or_fileobj=image_path,
            path_in_repo=os.path.basename(image_path),
            repo_id=repo_id,
            token=token,
            repo_type="dataset"
        )
        print(f"Uploaded {os.path.basename(image_path)}")
        time.sleep(1)  # Add delay to avoid rate limiting
        api.upload_file(
            path_or_fileobj=depth_map_path,
            path_in_repo=os.path.basename(depth_map_path),
            repo_id=repo_id,
            token=token,
            repo_type="dataset"
        )
        print(f"Uploaded {os.path.basename(depth_map_path)}")
        time.sleep(1)  # Add delay to avoid rate limiting

def process_and_upload(folder_path, model_name, repo_id):
    encoder = name2encoder[model_name]
    images, depth_maps, temp_dir = process_images(folder_path, encoder)
    upload_to_hf(images, depth_maps, repo_id)
    shutil.rmtree(temp_dir)  # Clean up temporary directory
    return f"Uploaded images and depth maps to {repo_id}"

def visualize_process(folder_path, model_name):
    encoder = name2encoder[model_name]
    images, depth_maps, temp_dir = process_images(folder_path, encoder)
    visualization = []
    for image_path, depth_map_path in zip(images, depth_maps):
        image = Image.open(image_path)
        depth_map = Image.open(depth_map_path)
        visualization.append([image, depth_map])
    shutil.rmtree(temp_dir)  # Clean up temporary directory
    return visualization

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🩻 Depth Map Generation and Upload to Hugging Face Hub 🩻")
    
    folder_input = gr.Textbox(label="Folder Path", placeholder="Enter the path to the folder with images")
    model_dropdown = gr.Dropdown(
        choices=["Small", "Base", "Large"],
        value="Small",
        label="Select Model Size"
    )
    repo_id_input = gr.Textbox(label="Hugging Face Repo ID", placeholder="Enter your Hugging Face repo ID")
    output = gr.Textbox(label="Output")
    visualization_output = gr.Gallery(label="Visualization of Generated Depth Maps", columns=2)
    
    process_button = gr.Button("Process and Upload")
    visualize_button = gr.Button("Visualize Process")
    
    process_button.click(fn=process_and_upload, inputs=[folder_input, model_dropdown, repo_id_input], outputs=output)
    visualize_button.click(fn=visualize_process, inputs=[folder_input, model_dropdown], outputs=visualization_output)

if __name__ == "__main__":
    demo.launch()