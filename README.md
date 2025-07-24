# 🔄 Bidirectional Real-Time Speech Transcription & Translation System

An advanced, fully offline real-time speech transcription system optimized for Hinglish and Indian accents with **bidirectional translation** and multi-engine text-to-speech capabilities.

## ✨ Features

### 🎯 Core Capabilities
- **Real-time Speech Transcription** using OpenAI Whisper (optimized for Hinglish/Indian accents)
- **Bidirectional Translation** with offline Argos Translate models
- **Advanced Multi-Engine TTS** (Microsoft Edge TTS, Google TTS, offline pyttsx3)
- **Language Detection** with automatic routing
- **Smart Audio Processing** with VAD and noise filtering

### 🔄 Translation Directions

#### Forward Translation (Indian Languages → English)
- **Hindi** → English
- **Tamil** → English  
- **Marathi** → English (with Hindi TTS fallback)

#### Reverse Translation (English → Indian Languages)
- **English** → Hindi (default)
- **English** → Tamil (configurable)

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For better audio support (optional)
# Download and install ffmpeg from: https://ffmpeg.org/download.html
```

### Basic Usage

```bash
# Default mode (English → Hindi)
python main.py

# Specify target language for English input
python main.py --target_language ta  # English → Tamil
python main.py --target_language hi  # English → Hindi

# Advanced options
python main.py --model small --device cuda --target_language ta
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --model {tiny,base,small,medium,large}    Whisper model (default: base)
  --device {cpu,cuda,mps,auto}              Computation device (default: auto)
  --target_language {hi,ta}                 Target for English input (default: hi)
  --energy_threshold INT                    Mic sensitivity (default: 1000)
  --confidence_threshold FLOAT              Transcription confidence (default: 0.3)
  --record_timeout FLOAT                    Recording duration (default: 2.0)
  --phrase_timeout FLOAT                    Phrase completion timeout (default: 3.0)
```

## 🎤 How It Works

### Bidirectional Pipeline

1. **Audio Capture** → Microphone input with noise filtering
2. **Speech Recognition** → Whisper transcription with confidence scoring
3. **Language Detection** → Automatic language identification
4. **Smart Translation** → Bidirectional routing:
   - **Hindi/Tamil/Marathi** detected → Translate to **English** → Speak in **English**
   - **English** detected → Translate to **Target Language** → Speak in **Target Language**
5. **Text-to-Speech** → Multi-engine TTS with language-specific voices

### Translation Matrix

| Input Language | Output Language | TTS Language | Status |
|----------------|-----------------|--------------|---------|
| Hindi (hi)     | English (en)    | English      | ✅ Active |
| Tamil (ta)     | English (en)    | English      | ✅ Active |
| Marathi (mr)   | English (en)    | English      | ✅ Active |
| English (en)   | Hindi (hi)      | Hindi        | ✅ Active |
| English (en)   | Tamil (ta)      | Tamil        | ✅ Active |

## 🎛️ Advanced Configuration

### TTS Engine Selection
The system automatically selects the best available TTS engine:
1. **Microsoft Edge TTS** (preferred, high quality)
2. **Google TTS** (fallback, requires internet)
3. **pyttsx3** (offline fallback)

### Model Recommendations
- **tiny**: Fastest, lower accuracy
- **base**: Balanced performance (recommended)
- **small**: Better accuracy, moderate speed
- **medium/large**: Highest accuracy, slower

### Device Selection
- **auto**: Automatically selects best available (CUDA > MPS > CPU)
- **cuda**: NVIDIA GPU acceleration
- **mps**: Apple Silicon acceleration
- **cpu**: CPU processing

## 📊 Real-Time Interface

```
🎙️ BIDIRECTIONAL REAL-TIME TRANSCRIPTION & TTS
=================================================================
Model: base | Device: cuda
🔄 Translation: Hindi/Tamil/Marathi → English | English → Hindi
⏱️ Runtime: 0:02:15 | Chunks: 23 | Avg Conf: 0.87
-----------------------------------------------------------------
[HI] नमस्ते, आप कैसे हैं? → (EN: Hello, how are you?)
[EN] Good morning! → (HI: सुप्रभात!)
[TA] வணக்கம் → (EN: Hello)
🎤 Current input...
─────────────────────────────────────────────────────────────────
Press Ctrl+C to stop and save transcript
```

## 📁 Output Files

Transcripts are automatically saved to `transcripts/` directory:
- **Format**: `transcript_YYYY-MM-DD_HH-MM-SS.txt`
- **Content**: Timestamped transcriptions with translations and confidence scores

## 🔧 Troubleshooting

### Common Issues

**ffmpeg Warning**: Install ffmpeg for better audio format support
```bash
# Windows: Download from https://ffmpeg.org/download.html
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
```

**Translation Models Missing**: Models download automatically on first run
```bash
# Manual installation if needed
python -c "from main import install_translation_models; install_translation_models()"
```

**Microphone Issues**: 
```bash
# List available microphones (Linux)
python main.py --default_microphone list
```

## 🎯 Use Cases

- **Live Meetings**: Real-time translation for multilingual participants
- **Language Learning**: Practice pronunciation with instant feedback
- **Accessibility**: Voice translation for hearing-impaired users
- **Content Creation**: Multilingual video/podcast production
- **Customer Service**: Real-time support in multiple languages

## 🔮 Future Enhancements

- [ ] Web interface for remote access
- [ ] Additional Indian languages (Bengali, Gujarati, Punjabi)
- [ ] Voice cloning for consistent speaker identity
- [ ] Real-time streaming to external applications
- [ ] Custom vocabulary and domain adaptation

## 📄 License

This project is open source and available under the MIT License.

---

**Made with ❤️ for the Indian community**
