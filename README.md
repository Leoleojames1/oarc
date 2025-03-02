# OARC (ollamaAgentRollCage)

OARC is an open-source toolkit for building Ollama-powered speech-enabled AI agents with vision capabilities.

## Prerequisites
- Python 3.8+
- Ollama installed and running
- PyAudio dependencies (portaudio)

## Installation
```bash
# Install PyAudio dependencies (Windows)
pip install pipwin
pipwin install pyaudio

# Install package
pip install oarc
```

## Quick Start

```python
from oarc import oarcAPI
from oarc.speechToSpeech import textToSpeech, speechToText

# Initialize API
api = oarcAPI()

# Use speech components
stt = speechToText()
tts = textToSpeech()

# Start listening
text = stt.recognizer(audio_file)

# Generate speech
tts.generate_audio("Hello, I am an Ollama AI assistant!")
```

## Advanced
```python
# Example usage
from oarc import oarcAPI
from oarc.speechToSpeech import textToSpeech
from oarc.promptModel import multiModalPrompting
from oarc.yoloProcessor import yoloProcessor

# Initialize components
api = oarcAPI()
tts = textToSpeech(developer_tools_dict={...})
vision = yoloProcessor()
llm = multiModalPrompting()

# Process speech and vision
text = api.recognize_speech(audio_file)
vision_data = vision.process_image(image_file)
response = llm.send_prompt(f"{text} {vision_data}")
audio = tts.generate_audio(response)
```

## Features

- Speech-to-Text with Whisper and Google Speech Recognition
- Text-to-Speech with Coqui TTS
- Vision AI with YOLO
- LLM integration with Ollama
- Conversation history with PandasDB
- WebSocket support for real-time audio

# To publish

# Build package
python -m pip install --upgrade build
python -m build

# Upload to PyPI
python -m pip install --upgrade twine
python -m twine upload dist/*