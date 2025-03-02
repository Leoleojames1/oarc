import os
import shutil
import time
import git
import pymongo
import logging
from typing import Dict

class GitHubRepoCloner:
    def __init__(self, storage_type: str = "mongodb", database_path: str = "mongodb://localhost:27017/"):
        """
        Initialize the GitHub Repo Cloner.
        
        Args:
            storage_type: Type of storage ("mongodb")
            database_path: Path to the database file or MongoDB URI
        """
        self.storage_type = storage_type
        self.database_path = database_path
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the cloner."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def clone_and_store_repo(self, agent_id: str, repo_url: str) -> None:
        """Clone a Git repository and store the code in the agent's code database."""
        try:
            # Define the path for the temporary clone
            temp_dir = f"/tmp/{agent_id}_repo"
            
            # Clone the repository
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            git.Repo.clone_from(repo_url, temp_dir)
            
            # Initialize the code database collection
            client = pymongo.MongoClient(self.database_path)
            db = client.agentCores
            code_collection_name = f"code_clone_{agent_id}"
            code_collection = db[code_collection_name]
            
            # Iterate through the cloned repository and store files
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    relative_path = os.path.relpath(file_path, temp_dir)
                    code_collection.insert_one({
                        "file_path": relative_path,
                        "content": content,
                        "last_updated": time.time(),
                        "repo_url": repo_url
                    })
            
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
            
            self.logger.info(f"Successfully cloned and stored repository {repo_url} for agent {agent_id}")
        except Exception as e:
            self.logger.error(f"Error cloning and storing repository {repo_url} for agent {agent_id}: {e}")
            raise