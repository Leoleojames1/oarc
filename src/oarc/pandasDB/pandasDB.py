"""pandasDB.py

please help me create a pandas db using the following colab code snippet for the rest of oarc: ollamaAgentRollCage
Pandas Query Engine
This guide shows you how to use our PandasQueryEngine: convert natural language to Pandas python code using LLMs.
"""

#TODO REFACTOR INTO A CLASS
import logging
import sys
from IPython.display import Markdown, display
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class PandasDB:
    def __init__(self):
        pass
    
    def chatbotPandasDB():
        #TODO CREATE AGENT CORES STORAGE FACILITY FOR OLLAMAAGENTROLLCAGE AGENTS 
        # CHANGE ALL OF THIS EXAMPLE CODE FOR LLAMA INDEX PANDAS QUERY ENGINE
        
        # Test on some sample data
        df = pd.DataFrame(
            {
                "city": ["Toronto", "Tokyo", "Berlin"],
                "population": [2930000, 13960000, 3645000],
            }
        )

        query_engine = PandasQueryEngine(df=df, verbose=True)
        response = query_engine.query(
            "What is the city with the highest population?",
        )

        display(Markdown(f"<b>{response}</b>"))

        # get pandas python instructions
        print(response.metadata["pandas_instruction_str"])

        query_engine = PandasQueryEngine(df=df, verbose=True, synthesize_response=True)
        response = query_engine.query(
            "What is the city with the highest population? Give both the city and population",
        )
        print(str(response))

        df = pd.read_csv("./titanic_train.csv")

        query_engine = PandasQueryEngine(df=df, verbose=True)

        response = query_engine.query(
            "What is the correlation between survival and age?",
        )

        display(Markdown(f"<b>{response}</b>"))

        # get pandas python instructions
        print(response.metadata["pandas_instruction_str"])

        from llama_index.core import PromptTemplate

        query_engine = PandasQueryEngine(df=df, verbose=True)
        prompts = query_engine.get_prompts()
        print(prompts["pandas_prompt"].template)

        print(prompts["response_synthesis_prompt"].template)

        new_prompt = PromptTemplate(
            """
            You are working with a pandas dataframe in Python.
            The name of the dataframe is `df`.
            This is the result of `print(df.head())`:
            {df_str}

            Follow these instructions:
            {instruction_str}
            Query: {query_str}

            Expression: """
            )

        query_engine.update_prompts({"pandas_prompt": new_prompt})


        instruction_str = """\
            1. Convert the query to executable Python code using Pandas.
            2. The final line of code should be a Python expression that can be called with the `eval()` function.
            3. The code should represent a solution to the query.
            4. PRINT ONLY THE EXPRESSION.
            5. Do not quote the expression.
            """
        return
    
    def storeAgent(self):
        """a method to store the current agent json config in the pandas db
        
        allows users to store the models associated with the agent
        """ 
        
    def load_from_json(self, load_name, large_language_model):
        """Load conversation history from JSON"""
        # Update load name and model
        self.load_name = load_name if load_name else f"conversation_{self.agent_id}_{self.current_date}"
        self.large_language_model = large_language_model
        
        # Update paths with new load name
        temp_save_name = self.save_name
        self.save_name = self.load_name
        self.updateConversationPaths()
        
        # Load conversation
        try:
            with open(self.pathLibrary['default_conversation_path'], 'r') as json_file:
                self.chat_history = json.load(json_file)
            print(f"Conversation loaded from: {self.pathLibrary['default_conversation_path']}")
            
            # Load from PandasDB
            self.chat_history = asyncio.run(self.conversation_handler.load_conversation(self.save_name))
            
        except Exception as e:
            print(f"Error loading conversation: {e}")
        finally:
            # Restore original save name
            self.save_name = temp_save_name
            self.updateConversationPaths()
            
    def save_to_json(self, save_name, large_language_model):
        """Save conversation history to JSON"""
        # Update save name and model
        self.save_name = save_name if save_name else f"conversation_{self.agent_id}_{self.current_date}"
        self.large_language_model = large_language_model
        
        # Update paths with new save name
        self.updateConversationPaths()
        
        # Save conversation
        try:
            with open(self.pathLibrary['default_conversation_path'], 'w') as json_file:
                json.dump(self.chat_history, json_file)
            print(f"Conversation saved to: {self.pathLibrary['default_conversation_path']}")
            
            # Save to PandasDB
            asyncio.run(self.conversation_handler.save_conversation(self.save_name, self.agent_id))
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
        
    def file_name_conversation_history_filter(self, input):
        # Use regex to replace all spaces with underscores and convert to lowercase
        output = re.sub(' ', '_', input).lower()
        return output
    
    def setAgent(self, agent_id):
        """Load agent configuration from agentCores and update current state"""
        try:
            logger.info("Loading agent configuration for agent_id: %s", agent_id)
            # From agent cores, load the agent configuration for the selected agent_id
            agent_config = self.agent_cores.loadAgentCore(agent_id)
            # set id in config
            agent_config["agentCore"]["agent_id"] = agent_id 
            
            # Remove the redundant agent_id key if it exists
            if 'agent_id' in agent_config:
                del agent_config['agent_id']
            
            if not agent_config:
                raise ValueError(f"Agent {agent_id} not found")

            # Ensure models key is present
            if "models" not in agent_config:
                agent_config["agentCore"]["models"] = {
                    "largeLanguageModel": {"names": [], "instances": [], "model_config_template": {}},
                    "embedding": {"names": [], "instances": [], "model_config_template": {}},
                    "largeLanguageAndVisionAssistant": {"names": [], "instances": [], "model_config_template": {}},
                    "yoloVision": {"names": [], "instances": [], "model_config_template": {}},
                    "speechRecognitionSTT": {"names": [], "instances": [], "model_config_template": {}},
                    "voiceGenerationTTS": {"names": [], "instances": [], "model_config_template": {}}
                }
                logger.info("Added missing 'models' key to agent_config")

            # Ensure prompts key is present
            if "prompts" not in agent_config:
                agent_config["agentCore"]["prompts"] = {
                    "userInput": "",
                    "llmSystem": "",
                    "llmBooster": "",
                    "visionSystem": "",
                    "visionBooster": "",
                    "primeDirective": ""
                }
                logger.info("Added missing 'prompts' key to agent_config")

            # Ensure modalityFlags key is present
            if "modalityFlags" not in agent_config:
                agent_config["agentCore"]["modalityFlags"] = {
                    "TTS_FLAG": False,
                    "STT_FLAG": False,
                    "CHUNK_AUDIO_FLAG": False,
                    "AUTO_SPEECH_FLAG": False,
                    "LLAVA_FLAG": False,
                    "SCREEN_SHOT_FLAG": False,
                    "SPLICE_VIDEO_FLAG": False,
                    "AUTO_COMMANDS_FLAG": False,
                    "CLEAR_MEMORY_FLAG": False,
                    "ACTIVE_AGENT_FLAG": False,
                    "EMBEDDING_FLAG": False,
                    "LLM_SYSTEM_PROMPT_FLAG": False,
                    "LLM_BOOSTER_PROMPT_FLAG": False,
                    "VISION_SYSTEM_PROMPT_FLAG": False,
                    "VISION_BOOSTER_PROMPT_FLAG": False
                }
                logger.info("Added missing 'modalityFlags' key to agent_config")

            # Ensure conversation key is present
            if "conversation" not in agent_config:
                agent_config["agentCore"]["conversation"] = {
                    "save_name": "",
                    "load_name": ""
                }
                logger.info("Added missing 'conversation' key to agent_config")

            # Update agent state from configuration
            self.agent_id = agent_config["agentCore"]["agent_id"]
            # models
            self.large_language_model = agent_config["agentCore"]["models"]["largeLanguageModel"]["names"][0] if agent_config["agentCore"]["models"]["largeLanguageModel"]["names"] else None
            self.embedding_model = agent_config["agentCore"]["models"]["embedding"]["names"][0] if agent_config["agentCore"]["models"]["embedding"]["names"] else None
            self.language_and_vision_model = agent_config["agentCore"]["models"]["largeLanguageAndVisionAssistant"]["names"][0] if agent_config["agentCore"]["models"]["largeLanguageAndVisionAssistant"]["names"] else None
            self.yolo_model = agent_config["agentCore"]["models"]["yoloVision"]["names"][0] if agent_config["agentCore"]["models"]["yoloVision"]["names"] else None
            self.whisper_model = agent_config["agentCore"]["models"]["speechRecognitionSTT"]["names"][0] if agent_config["agentCore"]["models"]["speechRecognitionSTT"]["names"] else None
            self.voice_name = agent_config["agentCore"]["models"]["voiceGenerationTTS"]["names"][0] if agent_config["agentCore"]["models"]["voiceGenerationTTS"]["names"] else None
            self.voice_type = agent_config["agentCore"]["models"]["voiceGenerationTTS"]["model_config_template"].get("voice_type", None)
            # prompts
            self.user_input_prompt = agent_config["agentCore"]["prompts"]["userInput"]
            self.llmSystemPrompt = agent_config["agentCore"]["prompts"]["llmSystem"]
            self.llmBoosterPrompt = agent_config["agentCore"]["prompts"]["llmBooster"]
            self.visionSystemPrompt = agent_config["agentCore"]["prompts"]["visionSystem"]
            self.visionBoosterPrompt = agent_config["agentCore"]["prompts"]["visionBooster"]
            # flags
            self.LLM_SYSTEM_PROMPT_FLAG = agent_config["agentCore"]["modalityFlags"]["LLM_SYSTEM_PROMPT_FLAG"]
            self.LLM_BOOSTER_PROMPT_FLAG = agent_config["agentCore"]["modalityFlags"]["LLM_BOOSTER_PROMPT_FLAG"]
            self.VISION_SYSTEM_PROMPT_FLAG = agent_config["agentCore"]["modalityFlags"]["VISION_SYSTEM_PROMPT_FLAG"]
            self.VISION_BOOSTER_PROMPT_FLAG = agent_config["agentCore"]["modalityFlags"]["VISION_BOOSTER_PROMPT_FLAG"]
            self.TTS_FLAG = agent_config["agentCore"]["modalityFlags"]["TTS_FLAG"]
            self.STT_FLAG = agent_config["agentCore"]["modalityFlags"]["STT_FLAG"]
            self.CHUNK_FLAG = agent_config["agentCore"]["modalityFlags"]["CHUNK_AUDIO_FLAG"]
            self.AUTO_SPEECH_FLAG = agent_config["agentCore"]["modalityFlags"]["AUTO_SPEECH_FLAG"]
            self.LLAVA_FLAG = agent_config["agentCore"]["modalityFlags"]["LLAVA_FLAG"]
            self.SCREEN_SHOT_FLAG = agent_config["agentCore"]["modalityFlags"]["SCREEN_SHOT_FLAG"]
            self.SPLICE_FLAG = agent_config["agentCore"]["modalityFlags"]["SPLICE_VIDEO_FLAG"]
            self.AGENT_FLAG = agent_config["agentCore"]["modalityFlags"]["ACTIVE_AGENT_FLAG"]
            self.MEMORY_CLEAR_FLAG = agent_config["agentCore"]["modalityFlags"]["CLEAR_MEMORY_FLAG"]
            self.EMBEDDING_FLAG = agent_config["agentCore"]["modalityFlags"]["EMBEDDING_FLAG"]
            # conversation metadata
            self.save_name = agent_config["agentCore"]["conversation"]["save_name"]
            self.load_name = agent_config["agentCore"]["conversation"]["load_name"]

            # Update paths
            self.updateConversationPaths()
            logger.info(f"Agent {agent_id} loaded successfully:\n%s", pformat(agent_config, indent=2, width=80))

        except Exception as e:
            logger.error(f"Error loading agent {agent_id}: {e}")
            raise
            
    async def list_available_agents(self) -> list:
        """Get list of available agents with details."""
        try:
            agents = self.agent_cores.listAgentCores()
            formatted_agents = []
            
            for agent in agents:
                # Load full config to get additional details
                config = self.agent_cores.loadAgentCore(agent["agent_id"])
                if config:
                    models = config["models"]
                    formatted_agents.append({
                        "id": agent["agent_id"],
                        "llm": models.get("largeLanguageModel", {}).get("names", [None])[0],
                        "vision": models.get("largeLanguageAndVisionAssistant", {}).get("names", [None])[0],
                        "voice": models.get("voiceGenerationTTS", {}).get("names", [None])[0],
                        "version": agent.get("version", "Unknown")
                    })
            
            return formatted_agents
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return []

    def create_agent_from_template(self, template_name: str, agent_id: str, custom_config: Optional[Dict] = None):
        """Create a new agent from a template"""
        try:
            # Use agentCores to create the agent from the template
            self.agent_cores.create_agent_from_template(template_name, agent_id)
            
            # Initialize the new agent
            self.setAgent(agent_id)
            logger.info(f"Agent {agent_id} created successfully from template {template_name}")
        except Exception as e:
            logger.error(f"Error creating agent from template: {e}")
            raise

    def save_agent_state(self):
        try:
            # Create current state configuration matching agentCore structure
            current_state = {
                "agentCore": {
                    "identifyers": {
                        "agent_id": self.agent_id,
                        "creationDate": self.current_date
                    },
                    "models": {
                        "largeLanguageModel": {
                            "names": [self.large_language_model] if self.large_language_model else [],
                            "instances": []
                        },
                        "embedding": {
                            "names": [self.embedding_model] if self.embedding_model else [],
                            "instances": []
                        },
                        "largeLanguageAndVisionAssistant": {
                            "names": [self.language_and_vision_model] if self.language_and_vision_model else [],
                            "instances": []
                        },
                        "yoloVision": {
                            "names": [self.yolo_model] if self.yolo_model else [],
                            "instances": []
                        },
                        "speechRecognitionSTT": {
                            "names": [self.whisper_model] if self.whisper_model else [],
                            "instances": []
                        },
                        "voiceGenerationTTS": {
                            "names": [self.voice_name] if self.voice_name else [],
                            "model_config_template": {
                                "voice_type": self.voice_type
                            }
                        }
                    },
                    "prompts": {
                        "userInput": self.user_input_prompt,
                        "agent": {
                            "llmSystem": self.llmSystemPrompt,
                            "llmBooster": self.llmBoosterPrompt,
                            "visionSystem": self.visionSystemPrompt,
                            "visionBooster": self.visionBoosterPrompt
                        }
                    },
                    "modalityFlags": {
                        "TTS_FLAG": self.TTS_FLAG,
                        "STT_FLAG": self.STT_FLAG,
                        "CHUNK_AUDIO_FLAG": self.CHUNK_FLAG,
                        "AUTO_SPEECH_FLAG": self.AUTO_SPEECH_FLAG,
                        "LLAVA_FLAG": self.LLAVA_FLAG,
                        "SCREEN_SHOT_FLAG": self.SCREEN_SHOT_FLAG,
                        "SPLICE_VIDEO_FLAG": self.SPLICE_FLAG,
                        "AUTO_COMMANDS_FLAG": False,
                        "CLEAR_MEMORY_FLAG": self.MEMORY_CLEAR_FLAG,
                        "ACTIVE_AGENT_FLAG": self.AGENT_FLAG,
                        "EMBEDDING_FLAG": self.EMBEDDING_FLAG,
                        "LLM_SYSTEM_PROMPT_FLAG": self.LLM_SYSTEM_PROMPT_FLAG,
                        "LLM_BOOSTER_PROMPT_FLAG": self.LLM_BOOSTER_PROMPT_FLAG,
                        "VISION_SYSTEM_PROMPT_FLAG": self.VISION_BOOSTER_PROMPT_FLAG,
                        "VISION_BOOSTER_PROMPT_FLAG": self.VISION_BOOSTER_PROMPT_FLAG

                    }
                }
            }
            self.agent_cores.storeAgentCore(self.agent_id, current_state)
            logger.info(f"Saved agent state: {self.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
            return False
        
    def coreAgent(self):
        # Define model configurations
        models_config = {
            "largeLanguageModel": {
                "names": [self.large_language_model] if self.large_language_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "embedding": {
                "names": [self.embedding_model] if self.embedding_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "largeLanguageAndVisionAssistant": {
                "names": [self.language_and_vision_model] if self.language_and_vision_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "yoloVision": {
                "names": [self.yolo_model] if self.yolo_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "speechRecognitionSTT": {
                "names": [self.whisper_model] if self.whisper_model else [],
                "instances": [],
                "model_config_template": {}
            },
            "voiceGenerationTTS": {
                "names": [self.voice_name] if self.voice_name else [],
                "instances": [],
                "model_config_template": {
                    "voice_type": self.voice_type
                }
            }
        }

        # Define prompt configurations
        prompts_config = {
            "userInput": self.user_input_prompt,
            "llmSystem": "",
            "llmBooster": "",
            "visionSystem": "",
            "visionBooster": "",
            "primeDirective": ""
        }

        # Define database configurations
        # TODO FIX HOME AGENT MATRIX NAME
        #TODO CONVERT TO PANDAS DB WITH JSON DATA FOR AGENTS
        database_config = {
            "agents": {
                "conversation": f"conversation_{self.agent_id}.db",
                "knowledge": f"knowledge_{self.agent_id}.db",
                "documentation": f"documentation_{self.agent_id}.db",
                "library": f"library_{self.agent_id}.db",
                "web_search": f"web_search_{self.agent_id}.db",
                "web_scrape": f"web_scrape_{self.agent_id}.db",
                "embeddings": f"embeddings_{self.agent_id}.db",
                "design_patterns": f"design_patterns_{self.agent_id}.db"
            },
        },

        # Define modality flags
        modality_flags = {
            "TTS_FLAG": self.TTS_FLAG,
            "STT_FLAG": self.STT_FLAG,
            "CHUNK_AUDIO_FLAG": self.CHUNK_FLAG,
            "AUTO_SPEECH_FLAG": self.AUTO_SPEECH_FLAG,
            "LLAVA_FLAG": self.LLAVA_FLAG,
            "SCREEN_SHOT_FLAG": self.SCREEN_SHOT_FLAG,
            "SPLICE_VIDEO_FLAG": self.SPLICE_FLAG,
            "AUTO_COMMANDS_FLAG": False,
            "CLEAR_MEMORY_FLAG": self.MEMORY_CLEAR_FLAG,
            "ACTIVE_AGENT_FLAG": self.AGENT_FLAG,
            "EMBEDDING_FLAG": self.EMBEDDING_FLAG,
            "LLM_SYSTEM_PROMPT_FLAG": False,
            "LLM_BOOSTER_PROMPT_FLAG": False,
            "VISION_SYSTEM_PROMPT_FLAG": False,
            "VISION_BOOSTER_PROMPT_FLAG": False
        }

        # Create the complete agent core structure
        self.agentCore = {
            "agentCore": {
                "identifyers": {
                    "agent_id": self.agent_id,
                    "uid": None,
                },
                "models": models_config,
                "prompts": prompts_config,
                "databases": database_config,
                "modalityFlags": modality_flags
            }
        }
        
    def initializeConversation(self):
        try:
            self.save_name = f"conversation_{self.agent_id}_{self.current_date}"
            self.load_name = self.save_name
            
            # Update conversation paths after initializing defaults
            self.updateConversationPaths()
            logger.info("Conversation initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            raise
        
    def updateConversationPaths(self):
        """Update conversation-specific paths"""
        try:
            agent_conversation_dir = os.path.join(self.pathLibrary['conversation_library_dir'], self.agent_id)
            os.makedirs(agent_conversation_dir, exist_ok=True)
            logger.info("Conversation paths updated successfully.")
        except Exception as e:
            logger.error(f"Error updating conversation paths: {e}")
            raise

    async def store_message(self, role, content, session_id=None, metadata=None):
        try:
            document = await self.conversation_handler.store_message(role, content, session_id, metadata)
            logger.info(f"Stored message: {document}")
        except Exception as e:
            logger.error(f"Error storing message: {e}")
            raise

    async def get_conversation_history(self, session_id=None, limit=100):
        try:
            history = await self.conversation_handler.get_conversation_history(session_id, limit)
            logger.info(f"Retrieved conversation history: {history}")
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            raise

    async def save_conversation(self, save_name, session_id):
        try:
            await self.conversation_handler.save_conversation(save_name, session_id)
            logger.info(f"Saved conversation with name: {save_name}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            raise

    async def load_conversation(self, save_name):
        try:
            conversation = await self.conversation_handler.load_conversation(save_name)
            logger.info(f"Loaded conversation with name: {save_name}")
            return conversation
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            raise

    async def clear_history(self, session_id=None):
        try:
            await self.conversation_handler.clear_history(session_id)
            logger.info(f"Cleared conversation history for session_id: {session_id}")
        except Exception as e:
            logger.error(f"Error clearing conversation history: {e}")
            raise