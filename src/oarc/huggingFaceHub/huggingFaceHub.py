"""huggingFaceHub.py
"""

from huggingface_hub import (
    create_repo, 
    upload_file, 
    upload_folder,
    hf_hub_download,
    snapshot_download,
    HfApi
)
import os
from typing import Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)

class HuggingFaceHub:
    """Manages interactions with Hugging Face Hub for model and dataset management"""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize HuggingFaceHub manager
        
        Args:
            token (str, optional): HuggingFace API token
        """
        self.token = token
        self.api = HfApi()
        self.login()
        
    def login(self) -> bool:
        """Log into Hugging Face Hub with token"""
        try:
            if self.token:
                self.api.set_access_token(self.token)
            return True
        except Exception as e:
            logger.error(f"Failed to login: {e}")
            return False

    def download_model(self, repo_id: str, filename: Optional[str] = None) -> str:
        """Download model file or snapshot from Hub
        
        Args:
            repo_id (str): Repository ID (e.g. "tiiuae/falcon-7b")
            filename (str, optional): Specific file to download
            
        Returns:
            str: Path to downloaded file/folder
        """
        try:
            if filename:
                path = hf_hub_download(repo_id=repo_id, filename=filename)
            else:
                path = snapshot_download(repo_id)
            logger.info(f"Downloaded {repo_id} to {path}")
            return path
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    def upload_model(self, 
                    repo_id: str,
                    local_path: str,
                    repo_type: str = "model") -> bool:
        """Upload model files to Hub
        
        Args:
            repo_id (str): Repository ID to create/update
            local_path (str): Local file/folder path to upload
            repo_type (str): Repository type ("model", "dataset", "space")
            
        Returns:
            bool: Success status
        """
        try:
            # Create or ensure repo exists
            create_repo(repo_id=repo_id, repo_type=repo_type)
            
            # Upload file or folder
            if os.path.isfile(local_path):
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=os.path.basename(local_path),
                    repo_id=repo_id,
                )
            else:
                upload_folder(
                    folder_path=local_path,
                    repo_id=repo_id,
                    repo_type=repo_type
                )
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def get_model_list(self, filter_criteria: Optional[Dict] = None) -> List[Dict]:
        """Get list of models matching criteria
        
        Args:
            filter_criteria (dict, optional): Filtering parameters
            
        Returns:
            list: List of matching model metadata
        """
        try:
            models = self.api.list_models(filter=filter_criteria)
            return [model.to_dict() for model in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model_tags(self, repo_id: str) -> List[str]:
        """Get tags for a model
        
        Args:
            repo_id (str): Repository ID
            
        Returns:
            list: List of tags
        """
        try:
            model_info = self.api.model_info(repo_id)
            return model_info.tags
        except Exception as e:
            logger.error(f"Failed to get tags: {e}")
            return []

    def validate_model(self, repo_id: str, expected_files: List[str]) -> bool:
        """Validate model has required files
        
        Args:
            repo_id (str): Repository ID
            expected_files (list): List of required filenames
            
        Returns:
            bool: Whether validation passed
        """
        try:
            files = self.api.list_repo_files(repo_id)
            missing = set(expected_files) - set(files)
            if missing:
                logger.warning(f"Missing files: {missing}")
                return False
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def get_model_info(self, model_id: str) -> Dict:
        """Get detailed model information
        
        Args:
            model_id (str): Model ID to look up
            
        Returns:
            dict: Model metadata
        """
        try:
            info = self.api.model_info(model_id)
            return info.to_dict()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

    async def download_model_async(self, *args, **kwargs):
        """Async wrapper for download_model"""
        return await asyncio.to_thread(self.download_model, *args, **kwargs)

    async def upload_model_async(self, *args, **kwargs): 
        """Async wrapper for upload_model"""
        return await asyncio.to_thread(self.upload_model, *args, **kwargs)