"""flagManger.py

This module contains the FlagManager class, which is responsible for managing the flags 
that are used to control state machine for the oarc agents in the oarc wizard.
"""
class FlagManager:
    def __init__(self, agent_id, command_library, new_state, tts_processor_instance, 
                 speech_recognizer_instance, yolo_processor_instance, large_language_model, 
                 current_dir, parent_dir, speech_dir, recognize_speech_dir, generate_speech_dir, 
                 tts_voice_ref_wav_pack_path):
        
        self.agent_id = agent_id
        self.command_library = command_library
        self.new_state = new_state
        self.tts_processor_instance = tts_processor_instance
        self.speech_recognizer_instance = speech_recognizer_instance
        self.yolo_processor_instance = yolo_processor_instance
        self.large_language_model = large_language_model
        self.current_dir = current_dir
        self.parent_dir = parent_dir
        self.speech_dir = speech_dir
        self.recognize_speech_dir = recognize_speech_dir
        self.generate_speech_dir = generate_speech_dir
        self.tts_voice_ref_wav_pack_path = tts_voice_ref_wav_pack_path
        
        self.TTS_FLAG = False
        self.STT_FLAG = False
        self.LLAVA_FLAG = False
        self.AUTO_SPEECH_FLAG = False
        
    async def get_command_library(self):
        return list(self.command_library.keys())
    
    def get_agent_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "flags": {
                "TTS_FLAG": self.TTS_FLAG,
                "STT_FLAG": self.STT_FLAG,
                "LLAVA_FLAG": self.LLAVA_FLAG,
                "AUTO_SPEECH_FLAG": self.AUTO_SPEECH_FLAG
            },
            "voice": {
                "type": self.voice_type,
                "name": self.voice_name
            },
            "model": self.large_language_model
        }
     
    def update_state(self):
        if "flags" in self.new_state:
            for flag, value in self.new_state["flags"].items():
                if hasattr(self, flag):
                    setattr(self, flag, value)
                    
        if "voice" in self.new_state:
            self.set_voice(
                self.new_state["voice"].get("type"),
                self.new_state["voice"].get("name")
            )
            
        if "model" in self.new_state:
            self.set_model(self.large_language_model)
        
    def voice(self, flag):
        # Update TTS_FLAG
        self.TTS_FLAG = flag

        if self.TTS_FLAG:
            print("- text to speech activated -")
            print("🎙️ You can press shift+alt to interrupt speech generation. 🎙️")
            
            # Initialize TTS processor if not already initialized
            if not self.tts_processor_instance:
                self.tts_processor_instance = textToSpeech(
                    developer_tools_dict={
                        'current_dir': self.current_dir,
                        'parent_dir': self.parent_dir,
                        'speech_dir': self.speech_dir,
                        'recognize_speech_dir': self.recognize_speech_dir,
                        'generate_speech_dir': self.generate_speech_dir,
                        'tts_voice_ref_wav_pack_path_dir': self.tts_voice_ref_wav_pack_path
                    },
                    voice_type=self.voice_type,
                    voice_name=self.voice_name
                )
        else:
            print("- text to speech deactivated -")

        print(f"TTS_FLAG FLAG STATE: {self.TTS_FLAG}")
        return
    
    def speech(self, flag1, flag2):
        if flag1 and flag2 == True:
            print("🎙️ Press ctrl+shift to open mic, press ctrl+alt to close mic and recognize speech, then press shift+alt to interrupt speech generation. 🎙️")
            self.get_voice_selection()
            self.tts_processor_instance = self.instance_tts_processor(self.voice_type, self.voice_name)
            
        return
       
    def llava_flow(self, flag):
        self.LLAVA_FLAG = flag
        print(f"LLAVA_FLAG FLAG STATE: {self.LLAVA_FLAG}")
        return
    
    def voice_swap(self, voice_name_selection):
        # Search for the name after 'forward slash voice swap'
        print(f"Agent voice swapped to {voice_name_selection}")
        print(f"<<< USER >>> ")
        
        return
        
    def listen(self):
        if not self.STT_FLAG:
            self.STT_FLAG = True
            print("- speech to text activated -")
            print("🎙️ Press ctrl+shift to open mic, press ctrl+alt to close mic and recognize speech, then press shift+alt to interrupt speech generation. 🎙️")
        else:
            print("- speech to text deactivated -")

        return

    def auto_commands(self, flag):
        self.auto_commands_flag = flag
        print(f"auto_commands FLAG STATE: {self.auto_commands_flag}")
        return
    
    def wake_commands(self, flag):
        self.speech_recognizer_instance.use_wake_commands = flag
        print(f"use_wake_commands FLAG STATE: {self.speech_recognizer_instance.use_wake_commands}")
        return

    def yolo_state(self, flag):
        self.yolo_flag = flag
        print(f"use_wake_commands FLAG STATE: {self.yolo_flag}")
        return
    
    def get_available_voices(self):
        #TODO TODO MOVE TO TTS FILE AND CLEAN UP CHATBOT WIZARD
        # Get list of fine-tuned models
        fine_tuned_dir = f"{self.parent_dir}/AgentFiles/Ignored_TTS/"
        fine_tuned_voices = [d[8:] for d in os.listdir(fine_tuned_dir) if os.path.isdir(os.path.join(fine_tuned_dir, d)) and d.startswith("XTTS-v2_")]
        
        # Get list of voice reference samples
        reference_voices = [d for d in os.listdir(self.tts_voice_ref_wav_pack_path) if os.path.isdir(os.path.join(self.tts_voice_ref_wav_pack_path, d))]
        
        return fine_tuned_voices, reference_voices
    
    def get_voice_selection(self):
        print(f"<<< Available voices >>>")
        fine_tuned_voices, reference_voices = self.get_available_voices()
        all_voices = fine_tuned_voices + reference_voices
        for i, voice in enumerate(all_voices):
            print(f"{i + 1}. {voice}")
        
        while True:
            selection = input("Select a voice (enter the number): ")
            try:
                index = int(selection) - 1
                if 0 <= index < len(all_voices):
                    selected_voice = all_voices[index]
                    if selected_voice in fine_tuned_voices:
                        self.voice_name = selected_voice
                        self.voice_type = "fine_tuned"
                    else:
                        self.voice_name = selected_voice
                        self.voice_type = "reference"
                    return
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
             
    async def get_available_voices(self):
        fine_tuned_dir = f"{self.parent_dir}/AgentFiles/Ignored_TTS/"
        fine_tuned_voices = [d[8:] for d in os.listdir(fine_tuned_dir) if os.path.isdir(os.path.join(fine_tuned_dir, d)) and d.startswith("XTTS-v2_")]
        reference_voices = [d for d in os.listdir(self.tts_voice_ref_wav_pack_path) if os.path.isdir(os.path.join(self.tts_voice_ref_wav_pack_path, d))]
        return {"fine_tuned": fine_tuned_voices, "reference": reference_voices}
    
    def start_speech_recognition(self):
        self.speech_recognition_active = True
        self.STT_FLAG = True
        self.AUTO_SPEECH_FLAG = True
        if self.SILENCE_FILTERING_FLAG:
            self.speech_recognizer_instance.start_continuous_listening()
        else:
            self.speech_recognizer_instance.listen()
        return {"status": "Speech recognition started"}
        
    def stop_speech_recognition(self):
        self.speech_recognition_active = False
        self.STT_FLAG = False
        self.AUTO_SPEECH_FLAG = False
        self.speech_recognizer_instance.cleanup()
        return {"status": "Speech recognition stopped"}
    
    def toggle_speech_recognition(self):
        if self.speech_recognition_active:
            return self.stop_speech_recognition()
        else:
            return self.start_speech_recognition()
                
    def process_tts_and_send_audio(self, response):
        audio_data = self.tts_processor_instance.process_tts_responses(response, self.voice_name)
        if audio_data:
            # Send audio data to API endpoint for frontend visualization
            asyncio.run(self.send_audio_to_frontend(audio_data, "tts"))
        return audio_data
    
    async def send_audio_to_frontend(self, audio_data, audio_type):
        async with websockets.connect('ws://localhost:2020/audio-stream') as websocket:
            await websocket.send(json.dumps({
                'audio_type': audio_type,
                'audio_data': list(audio_data)
            }))
    
    def continuous_listening(self):
        if self.CONTINUOUS_LISTENING_FLAG:
            self.speech_recognizer_instance.start_continuous_listening()
        else:
            self.speech_recognizer_instance.listen()
            
    def interrupt_speech(self):
        if hasattr(self, 'tts_processor_instance'):
            self.tts_processor_instance.interrupt_generation()
    
    def get_user_audio_data(self):
        return self.speech_recognizer_instance.get_audio_data()
    
    def get_llm_audio_data(self):
        return self.tts_processor_instance.get_audio_data()
    
    def instance_tts_processor(self, voice_type, voice_name):
        try:
            if not hasattr(self, 'tts_processor_instance') or self.tts_processor_instance is None:
                self.tts_processor_instance = textToSpeech(self.pathLibrary, voice_type, voice_name)
            return self.tts_processor_instance
        except Exception as e:
            logger.error(f"Error instantiating TTS processor: {e}")
            return None
        
    def _initialize_conversation_handler_and_prompting(self):
        self.multi_modal_prompting = multiModalPrompting()
        logger.info("Multi-modal prompting initialized successfully.")
        

    def set_voice(self, voice_type: str, voice_name: str):
        try:
            if not voice_type or not voice_name:
                raise ValueError("Voice type and name must be provided")
                
            self.voice_type = voice_type
            self.voice_name = voice_name
            
            # Initialize TTS processor if needed
            if not hasattr(self, 'tts_processor_instance') or self.tts_processor_instance is None:
                self.tts_processor_instance = self.instance_tts_processor(voice_type, voice_name)
            else:
                self.tts_processor_instance.update_voice(voice_type, voice_name)
                
            self.TTS_FLAG = True  # Enable TTS when voice is set
            return True
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            return False
        
    def cleanup(self):
        try:
            if self.tts_processor_instance:
                self.tts_processor_instance.cleanup()
            if self.speech_recognizer_instance:
                self.speech_recognizer_instance.cleanup()
            # Clear chat history
            self.chat_history = []
            self.llava_history = []
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_chat_state(self):
        return {
            "chat_history": self.chat_history,
            "llava_history": self.llava_history,
            "current_model": self.large_language_model,
            "voice_state": {
                "type": self.voice_type,
                "name": self.voice_name,
                "tts_enabled": self.TTS_FLAG,
                "stt_enabled": self.STT_FLAG
            },
            "vision_state": {
                "llava_enabled": self.LLAVA_FLAG,
                "language_and_vision_model": getattr(self, 'language_and_vision_model', None)
            },
            "yolo_state": {
                "yolo_enabled": self.YOLO_FLAG,
                "yolo_model": getattr(self, 'yolo_model_name', None),
                "yolo_json_payload": getattr(self, 'yolo_json_payload', None)
            },
        }
