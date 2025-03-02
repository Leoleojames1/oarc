import cv2
import numpy as np
import torch
import queue
import threading
import sounddevice as sd
import webrtcvad
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pyvirtualcam
from pyvirtualcam import PixelFormat
from collections import deque
import time

class WhisperWebcam:
    def __init__(self, sample_rate=16000, frame_duration=30):
        # List available audio devices and let user select
        print("\nAvailable audio input devices:")
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}")
                input_devices.append(i)
        
        if not input_devices:
            raise ValueError("No input devices found!")
        
        # Let user select device
        while True:
            try:
                device_id = int(input("Select input device number: "))
                if device_id in input_devices:
                    break
                print("Invalid device number. Please try again.")
            except ValueError:
                print("Please enter a number.")
        
        self.audio_device = device_id
        print(f"Using audio device: {devices[device_id]['name']}")
        
        # Audio settings
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # ms
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)
        self.audio_queue = queue.Queue()
        self.captions = deque(maxlen=3)  # Store last 3 captions
        
        # Initialize Whisper
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {self.device}")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print("Loading Whisper model...")
        model_id = "distil-whisper/distil-large-v3"
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(self.device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs={"max_new_tokens": 128}
        )
        
        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self._audio_capture, daemon=True)
        self.running = True
        self.audio_thread.start()
        
        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.transcription_thread.start()
    
    def _audio_capture(self):
        """Capture audio from microphone with silence detection"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            
            # Ensure single channel by taking first channel if multiple channels
            if indata.shape[1] > 1:
                audio_data = indata[:, 0]
            else:
                audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
            
            # Convert audio to the format expected by WebRTC VAD
            audio_frame = (audio_data * 32767).astype(np.int16)
            
            # Check if frame contains speech
            try:
                frame_bytes = audio_frame.tobytes()
                if self.vad.is_speech(frame_bytes, self.sample_rate):
                    self.audio_queue.put(audio_frame)
            except Exception as e:
                print(f"VAD error: {e}")
        
        try:
            with sd.InputStream(
                device=self.audio_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=audio_callback,
                blocksize=int(self.sample_rate * self.frame_duration / 1000)
            ):
                print(f"Started audio capture on device {self.audio_device}")
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Audio capture error: {e}")
    
    def _process_audio(self):
        """Process audio chunks and perform transcription"""
        audio_buffer = []
        silence_threshold = 10  # Number of silent frames before processing
        silent_frames = 0
        
        while self.running:
            try:
                audio_frame = self.audio_queue.get(timeout=1.0)
                audio_buffer.append(audio_frame)
                silent_frames = 0
            except queue.Empty:
                silent_frames += 1
                
                # Process buffer after enough silence or if buffer is large enough
                if (silent_frames >= silence_threshold and audio_buffer) or len(audio_buffer) >= 50:
                    try:
                        # Concatenate audio frames
                        audio_data = np.concatenate(audio_buffer)
                        
                        # Ensure audio is single channel and properly shaped
                        if len(audio_data.shape) > 1:
                            audio_data = audio_data.reshape(-1)
                        
                        # Prepare input in the format expected by the pipeline
                        inputs = {
                            "raw": audio_data.astype(np.float32) / 32767.0,  # Convert back to float32
                            "sampling_rate": self.sample_rate
                        }
                        
                        # Perform transcription
                        result = self.pipe(inputs)
                        
                        if result["text"].strip():
                            self.captions.append(result["text"].strip())
                            print(f"Transcribed: {result['text'].strip()}")
                    
                    except Exception as e:
                        print(f"Transcription error: {e}")
                    
                    # Clear buffer
                    audio_buffer = []
    
    def draw_captions(self, frame):
        """Draw captions at the bottom of the frame"""
        height, width = frame.shape[:2]
        caption_height = 40
        spacing = 5
        
        # Create semi-transparent overlay for captions
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - (caption_height * len(self.captions))), 
                     (width, height), (0, 0, 0), -1)
        
        # Add captions
        for i, caption in enumerate(self.captions):
            y_pos = height - (caption_height * (len(self.captions) - i)) + spacing
            cv2.putText(overlay, caption, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Blend overlay with original frame
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame
    
    def virtual_webcam_stream(self):
        """Stream webcam with real-time captions to virtual camera"""
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise ValueError("Could not open webcam")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            with pyvirtualcam.Camera(width=width, height=height, fps=30, fmt=PixelFormat.BGR, backend='obs') as cam:
                print(f'Using virtual camera: {cam.device}')
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Add captions to frame
                    frame_with_captions = self.draw_captions(frame)
                    
                    # Send to virtual camera
                    cam.send(frame_with_captions)
                    cam.sleep_until_next_frame()
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
        except Exception as e:
            print(f"Virtual camera error: {e}")
        finally:
            self.running = False
            if 'cap' in locals():
                cap.release()

if __name__ == "__main__":
    try:
        whisper_cam = WhisperWebcam()
        whisper_cam.virtual_webcam_stream()
    except Exception as e:
        print(f"Main error: {e}")