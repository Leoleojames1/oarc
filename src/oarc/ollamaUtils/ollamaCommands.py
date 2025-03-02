""" ollama_commands.py

    This module contains the class ollama_commands which contains methods for interacting with the ollama library.
    
    @LBorcherding
"""
import ollama
import sys
import logging
from functools import partial
import asyncio

class ollamaCommands:
    def __init__(self):
        name = "ollamaCommands"
    
    def quit(self):
        sys.exit()

    async def ollama_show_modelfile(self, user_input_model_select):
        return ollama.show(f"{user_input_model_select}")
    
    async def ollama_show_template(self, user_input_model_select):
        modelfile_data = ollama.show(f"{user_input_model_select}")
        return modelfile_data.get('template', '')
    
    async def ollama_show_license(self, user_input_model_select):
        modelfile_data = ollama.show(f"{user_input_model_select}")
        return modelfile_data.get('license', '')

    async def ollama_show_loaded_models(self):
        ollama_loaded_models = ollama.ps()
        return ollama_loaded_models
    
    async def ollama_list(self):
        """Get list of available models"""
        try:
            logging.info("Calling ollama.list() to get available models")
            result = ollama.list()
            logging.info(f"ollama.list() result: {result}")
            if not isinstance(result, dict) or 'models' not in result:
                logging.warning("Unexpected response format from ollama.list()")
                return []
                
            # Extract and format model names
            models = [model['name'] for model in result['models'] if 'name' in model]
            logging.info(f"Extracted models: {models}")
            return models
            
        except Exception as e:
            logging.error(f"Error listing models: {str(e)}")
            return []
