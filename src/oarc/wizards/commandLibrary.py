"""commmandLibrary.py

This file contains the command library for the wizard. It is a dictionary of commands that the wizard can execute.
"""
class commandLibrary:
    def __init__(self):
        self.command_library = {}
        self.updateCommandLibrary()
        
    def updateCommandLibrary(self):
        self.command_library = {
            "swap": {
                "method": lambda swap_model_selection: self.swapClear(swap_model_selection),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /swap, changes the main llm model of the agent. This "
                    "command allows the user or the agent to swap in a new llm on the fly for intensive "
                    "agent modularity. "
                ),
            },
            "voice swap": {
                "method": lambda voice_name: self.voice_swap(voice_name),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /voice swap, swaps the current text to speech model out "
                    "for the specified voice name."
                ),
            },
            "agent select": {
                "method": lambda agent_id: self.setAgent(agent_id),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "Lists available agents in the agentCores matrix and allows selection. "
                    "Loads the selected agent's configuration including models, prompts, and flags."
                ),
            },
            "agent save": {
                "method": lambda: self.save_agent_state(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": "Save current agent state to agentCores"
            },
            "agent create": {
                "method": lambda template_name, agent_id: self.create_agent_from_template(template_name, agent_id),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": "Create a new agent from template"
            },
            "agent list": {
                "method": lambda: self.list_available_agents(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": "List all available agents"
            },
            "save as": {
                "method": lambda save_name, large_language_model: self.save_to_json(save_name, large_language_model),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /save as, allows the user to save the current conversation "
                    "history with the provided save name, allowing the conversation to be stored in a json. "
                ),
            },
            "load as": {
                "method": lambda load_name, large_language_model: self.load_from_json(load_name, large_language_model),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /load as, allows the user to provide the desired conversation "
                    "history which pulls from the conversation library, loading it into the agent allowing the "
                    "conversation to pick up where it left off. "
                ),
            },
            "write modelfile": {
                "method": lambda: self.model_write_class_instance.write_model_file(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /write modelfile, allows the user to design, customize, and build "
                    "their own modelfile for custom systemprompt loading, as well as gguf model selection, LoRA, adapter "
                    "merging, context length modification, as well as other ollama modelfile assets. For more Description "
                    "on ollama modelfiles check out the ollama documentation at: "
                    "https://github.com/ollama/ollama/blob/main/docs/modelfile.md "
                ),
                "documentation": "https://github.com/ollama/ollama/blob/main/docs/modelfile.md"
            },
            "convert tensor": {
                "method": lambda tensor_name: self.create_convert_manager_instance.safe_tensor_gguf_convert(tensor_name),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /convert tensor, allows the user to run the custom batch tool, "
                    "calling upon the llama.cpp repo for the convert_hf_to_gguf.py tool. For more information about "
                    "this llama.cpp tool, check out the following link to the documentation: "
                    "https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py"
                ),
                "documentation": "https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py"
            },
            "convert gguf": {
                "method": lambda STT_FLAG, model_git: self.model_write_class_instance.write_model_file_and_run_agent_create_gguf(STT_FLAG, model_git),
                "args": True,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /convert gguf, allows the user to convert any gguf model to an ollama model by constructing "
                    "the modelfile, and specifying the path to the gguf used for creating the model, in addition to other metadata."
                    "For more information you can check out the documentation at: "
                    "https://github.com/ollama/ollama/blob/main/docs/modelfile.md "
                ),
                "documentation": "https://github.com/ollama/ollama/blob/main/docs/modelfile.md",
            },
            "listen on": {
                "method": lambda: self.listen(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /listen on, changes the state of the listen flag & allows the " 
                    "user to activate the speech generation for the agent. "
                ),
            },
            "listen off": {
                "method": lambda: self.listen(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /listen off, changes the state of the listen flag & allows the " 
                    "user to deactivate the speech generation for the agent. "
                ),
            },
            "voice on": {
                "method": lambda: self.voice(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "the command, /voice on, changes the state of the voice flag," 
                    "in turn enabling the text to speech model in the agent."
                ),
            },
            "voice off": {
                "method": lambda: self.voice(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /voice off, changes the state of the voice flag," 
                    "in turn disabling the text to speech model in the agent."
                ),
            },
            "speech on": {
                "method": lambda: self.speech(True, True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /speech on, changes the state of the listen & voice "
                    "flags enabling speech recognition and speech generation for the agent."
                ),
            },
            "speech off": {
                "method": lambda: self.speech(False, False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /speech off, changes the state of the listen & voice "
                    "flags disabling speech recognition and speech generation for the agent. "
                ),
            },
            "wake on": {
                "method": lambda: self.setAgent(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /wake on, changes the state of the wake_flag, allowing the user "
                    "to enable wake names for the speech recognition, this can allow the agent to "
                    "be awoken with a phrase, and with advanced mode can respond to conversation "
                    "data said prior to the wake command through organized listening & chunk processing "
                    "of the user input audio in the past ~5 min cache, then sending this processed chunk "
                    "which had all silence removed, to the whisper speech to text model. "
                ),
            },
            "wake off": {
                "method": lambda: self.setAgent(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /wake on, changes the state of the wake_flag, allowing the user "
                    "to disable wake names for the speech recognition, this can allow the agent to "
                    "be awoken with a phrase, and with advanced mode can respond to conversation "
                    "data said prior to the wake command through organized listening & chunk processing "
                    "of the user input audio in the past ~5 min cache, then sending this processed chunk "
                    "which had all silence removed, to the whisper speech to text model. "
                ),
            },
            "latex on": {
                "method": lambda: self.setAgent(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /latex on, allows the user to activate the specilized latex rendering utility. "
                    "This is a specific rendering feature and is highly related to the system prompt, as well as "
                    "the artifact generation from the model output. Enabling this flag will allow for latex "
                    "mathematics rendering. "
                ),
            },
            "latex off": {
                "method": lambda: self.setAgent(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /latex off, allows the user to deactivate the specilized latex rendering utility. "
                    "This is a specific rendering feature and is highly related to the system prompt, as well as "
                    "the artifact generation from the model output. Enabling this flag will allow for latex "
                    "mathematics rendering. "
                ),
            },
            "command auto on": {
                "method": lambda: self.auto_commands(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /command auto on, allows the user to activate the auto commanding feature of the agent. "
                    "This feature enabled the ollama agent roll cage chatbot agent to project, infer, and execute commands in "
                    "the agent library automatically based on the user request speech data. Auto commands allows the agent to submit "
                    "/command prompts and command lists for tool execution. "
                ),
            },
            "command auto off": {
                "method": lambda: self.auto_commands(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /command auto off, allows the user to deactivate the auto commanding feature of the agent. "
                    "This feature disables the ollama agent roll cage chatbot agent to project, infer, and execute commands in "
                    "the agent library automatically based on the user request speech data. Auto commands allows the agent to submit "
                    "/command prompts and command lists for tool execution. "
                ),
            },
            "llava flow": {
                "method": lambda: self.llava_flow(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /llava flow, allows the user to activate the llava vision model in ollama, within the chatbot agent. "
                    "This is done through specialized a custom LLAVA_SYSTEM_PROMPT & LLAVA_BOOSTER_PROMPT, these prompts are provided in "
                    "The agent library. Once collected from the library the system & booster prompts are seeded in with the user speech "
                    "or text request to create llava vision prompts. "
                ),
            },
            "llava freeze":  {
                "method": lambda: self.llava_flow(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /llava freeze, allows the user to activate the llava vision model in ollama, within the chatbot agent. "
                    "This is done through specialized a custom LLAVA_SYSTEM_PROMPT & LLAVA_BOOSTER_PROMPT, these prompts are provided in "
                    "The agent library. Once collected from the library the system & booster prompts are seeded in with the user speech "
                    "or text request to create llava vision prompts. "
                ),
            },
            "yolo on": {
                "method": lambda: self.yolo_state(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": ( 
                    "The command, /yolo on, allows the user to activate Yolo real time object recognition model. Yolo stands for `You only "
                    "look once`. This model is able to provide bounding box data for objects on the computer screen, in the webcam, and more. "
                    "Activating yolo in the ollama agent roll cage chatbot agent framework, will allow the agent to utilizing Yolo data for "
                    "various agent frameworks. This includes the minecraft agent, the general navigator vision agent, the webcam ai chat, security "
                    "camera monitoring, and more, within the oarc environment. "
                ),
            },
            "yolo off": {
                "method": lambda: self.yolo_state(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /yolo off, allows the user to deactivate Yolo real time object recognition model. Yolo stands for `You only "
                    "look once`. This model is able to provide bounding box data for objects on the computer screen, in the webcam, and more. "
                    "Deactivating yolo in the ollama agent roll cage framework, will disallow the agent to utilizing Yolo data for "
                    "various agent frameworks. This includes the minecraft agent, the general navigator vision agent, the webcam ai chat, security "
                    "camera monitoring, and more, within the oarc environment. "
                ),
            },
            "auto speech on": {
                "method": lambda: self.auto_speech_set(True),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /auto speech on, allows the user to activate automatic speech to speech."
                ),
            },
            "auto speech off": {
                "method": lambda: self.auto_speech_set(False),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /auto speech on, allows the user to deactivate automatic speech to speech."
                ),
            },
            "quit": {
                "method": lambda: self.ollamaCommandInstance.quit(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /quit, allows the user to quit the ollama chatbot instance Shuting down "
                    "all chatbot agent processes."
                ),
            },
            "ollama create": {
                "method": lambda STT_FLAG: self.ollamaCommandInstance.ollama_create(),
                "args": False,
                "is_async": False,
                "LOCK": "STT_FLAG",
                "adminOnly": True,
                "description": (
                    "The command, /ollama create, allows the user run the ollama model creation command. Starting "
                    "the model creation menu, accepting the modelfile from /write modelfile. This will run the base "
                    "ollama create command with the specified arguments."
                    # TODO ADD LOCK ARG: ONLY RUN IN TEXT TO TEXT MODE
                    # IF LISTEN & LEAP ARE NOT DISABLED, NO OLLAMA CREATE
                    # TODO Add full speech lockdown commands, /quit, /stop, /freeze, /rewind <turns>, for spacial vision
                    # navigation and agentic action output spaces, such as robotics, voice commands, from admin users,
                    # who have been voice recognized as the correct person, these users can activate admin commands,
                    # to access lockdown protocols, since voice recognition is not full proof, this feature can
                    # be swapped in for a password, or a 2 factor authentification connected to an app on your phone.
                    # from there the admin control pannel voice commands, and buttons can be highly secure for
                    # admin personel only.
                    # TODO add encrypted speech and text output, allowing voice and text in, with encrypted packages.
                    # goal: encrypt speech to speech for interaction with the agent, but all output is garbled, this
                    # will act like a cipher, and only those with the key, or those who did the prompting will have
                    # access to. The general output of files, and actions will still be committed, and this essentially 
                    # lets you hide any one piece of information before deciding if you want to make it public with your
                    # decryption method and method of sharing and visualizing the chat.
                ),
            },
            "quit": {
                "method": lambda: self.ollamaCommandInstance.quit(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /quit, allows the user to quit the ollama chatbot instance Shuting down "
                    "all chatbot agent processes."
                ),
            },
            "ollama show": {
                "method": lambda: self.ollamaCommandInstance.ollama_show_modelfile(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "is_async": True,
                "description": (
                    "The command, /ollama show, allows the user to quit the ollama chatbot instance Shuting down "
                    "all chatbot agent processes."
                ),
            },
            "ollama template": {
                "method": lambda: self.ollamaCommandInstance.ollama_show_template(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /ollama template, displays the model template from the modelfile "
                    "for the currently loaded ollama llm in the chatbot agent. The template structure defines the llm "
                    "response patterns, and specifies the defined template for user, system, assistant roles, as well "
                    "as prompt structure. "
                ),
            },
            "ollama license": {
                "method": lambda: self.ollamaCommandInstance.ollama_show_license(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /ollama license, displays the license from the LLM modelfile of the current "
                    "model in the agent. This license comes from the distributor of the model and defines its usage "
                    "capabilities. "
                ),
            },
            "ollama list": {
                "method": lambda: self.ollamaCommandInstance.ollama_list(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /ollama list, displays the list of ollama models on the users machine, specificially "
                    "providing the response from the ollama list command through the ollama api. "
                ),
            },
            "ollama loaded": {
                "method": lambda: self.ollamaCommandInstance.ollama_show_loaded_models(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /ollama loaded, displayes all currently loaded ollama models. "
                    "This information is retrieved with the ollama.ps() method."
                ),
            },
            "splice video": {
                "method": lambda: self.data_set_video_process_instance.generate_image_data(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /splice video, splices the provided video into and image set that can be used for labeling. "
                    "Once this data is labeled in a tool such as Label Studio, it can be used for training Yolo, LlaVA and "
                    "other vision models. "
                ),
            },
            "start node": {
                "method": lambda: self.FileSharingNode_instance.start_node(),
                "args": False,
                "is_async": True,
                "adminOnly": True,
                "description": (
                    "The command, /start node, activates the peer-2-peer encrypted network node. This module "
                    "provides the necessary toolset for encrypted agent networking for various tasks. "
                ),
            },
            "conversation parquet": {
                "method": lambda: self.generate_synthetic_data(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /conversation parquet, converts the specified conversation name to a parquet dataset. "
                    "This dataset can be exported to huggingface for llm finetuning, and can be found in the conversation "
                    "history library under the parquetDatasets folder."
                ),
            },
            "convert wav": {
                "method": lambda: self.data_set_video_process_instance.call_convert(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /convert wav, calls the audio wav conversion tool. (WIP: may not be functioning)"
                ),
            },
            "shot prompt": {
                "method": lambda: self.shot_prompt(),
                "args": False,
                "is_async": False,
                "adminOnly": True,
                "description": (
                    "The command, /shot prompt, prompts the ollama model with the args following the command. "
                    "This prompt is done in a new conversation"
                ),
            },
        }