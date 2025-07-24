#!/usr/bin/env python3
"""
Advanced Real-Time Speech Transcription System
Optimized for Hinglish and Indian accents with enhanced performance and TTS
"""

import argparse
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Optional, Dict, Tuple
import warnings

import numpy as np
import torch
import whisper
from whisper.audio import SAMPLE_RATE
import speech_recognition as sr
from scipy import signal
import webrtcvad
import librosa
from langdetect import detect, LangDetectException

# Import the advanced TTS system
from tts import AdvancedTTS, TTSConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Language Detection and Translation ---

def install_translation_models():
    """Download and install Argos Translate models for supported languages."""
    try:
        import argostranslate.package
        import argostranslate.translate
        print("Checking for translation models...")
        required_languages = {"hi", "mr", "ta"}
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        installed_languages = {lang.code for lang in argostranslate.translate.get_installed_languages()}

        for lang_code in required_languages:
            if lang_code not in installed_languages:
                print(f"Downloading translation model for '{lang_code}'...")
                try:
                    package_to_install = next(
                        p for p in available_packages if p.from_code == lang_code and p.to_code == 'en'
                    )
                    package_to_install.install()
                    print(f"Model for '{lang_code}' installed successfully.")
                except StopIteration:
                    print(f"Warning: Could not find translation package for '{lang_code}' to 'en'.")
                except Exception as e:
                    print(f"Error installing model for '{lang_code}': {e}")
    except ImportError:
        print("Warning: argostranslate not found. Translation will be disabled.")
        print("Please install it with: pip install argostranslate")

def detect_language(text: str) -> str:
    """Detect the language of a given text."""
    if not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def translate_text(text: str, from_code: str, to_code: str = "en") -> str:
    """Translate text from a source language to a target language."""
    try:
        import argostranslate.translate
        translation = argostranslate.translate.translate(text, from_code, to_code)
        return translation
    except Exception as e:
        return f"Translation error: {e}"

class AudioProcessor:
    """Advanced audio processing for better transcription quality"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhanced audio preprocessing pipeline"""
        if audio_data.size == 0:
            return audio_data
        audio_data = audio_data / np.max(np.abs(audio_data) + 1e-8)
        sos = signal.butter(4, 80, btype='high', fs=self.sample_rate, output='sos')
        audio_data = signal.sosfilt(sos, audio_data)
        return audio_data.astype(np.float32)
    
    def detect_speech(self, audio_chunk: bytes) -> bool:
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception:
            return True

class TranscriptionBuffer:
    """Smart buffer management for smooth transcription"""
    
    def __init__(self, max_lines: int = 15):
        self.transcript = deque(maxlen=max_lines)
        self.current_line = ""
        
    def add_line(self, text: str, confidence: float, lang: str, translation: str):
        if text.strip():
            self.transcript.append((datetime.now(), text.strip(), lang, translation, confidence))
            self.current_line = ""
    
    def update_current(self, text: str):
        self.current_line = text.strip()
    
    def get_display_text(self) -> str:
        display_lines = []
        for _, line, lang, translation, _ in self.transcript:
            display_line = f"[{lang}] {line}"
            if translation:
                display_line += f" -> (EN: {translation})"
            display_lines.append(display_line)
        if self.current_line:
            display_lines.append(f" {self.current_line}")
        return "\n".join(display_lines)
    
    def export_transcript(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Real-Time Transcription Log\n")
            f.write("="*30 + "\n")
            for timestamp, line, lang, translation, confidence in self.transcript:
                f.write(f"[{timestamp.strftime('%H:%M:%S')}] [{lang}] (Conf: {confidence:.2f}) {line}")
                if translation:
                    f.write(f" -> (EN: {translation})")
                f.write("\n")

class SmartTranscriber:
    """Advanced transcription engine with Hinglish optimization"""
    
    def __init__(self, model_name: str, device: str):
        self.device = self._get_optimal_device(device)
        self.model = self._load_optimized_model(model_name)
        self.audio_processor = AudioProcessor()
        self.buffer = TranscriptionBuffer()
        
    def _get_optimal_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available(): return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return "mps"
        return device
    
    def _load_optimized_model(self, model_name: str):
        print(f"Loading {model_name} model on {self.device}...")
        model = whisper.load_model(model_name, device=self.device)
        if self.device == "cuda":
            model = model.half()
        model.eval()
        return model
    
    def transcribe_chunk(self, audio_data: np.ndarray) -> Tuple[str, float]:
        if audio_data.size < 100: return "", 0.0
        processed_audio = self.audio_processor.preprocess_audio(audio_data)
        try:
            result = self.model.transcribe(
                processed_audio, language="hi", task="transcribe", fp16=(self.device == "cuda"),
                temperature=0.2, no_speech_threshold=0.6,
                initial_prompt="This is a conversation in Hinglish (Hindi-English mix) with an Indian accent."
            )
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            if not segments:
                return "", 0.0
            confidence = np.mean([seg.get("no_speech_prob", 1.0) for seg in segments])
            confidence = 1 - confidence # Invert because it's no_speech_prob
            return text, confidence
        except Exception as e:
            print(f"Transcription error: {e}")
            return "", 0.0

class RealTimeTranscriber:
    """Main real-time transcription system"""
    
    def __init__(self, args):
        self.args = args
        self.transcriber = SmartTranscriber(args.model, args.device)
        self.data_queue = Queue()
        self.phrase_time = None
        self.last_sample = bytes()
        self.setup_microphone()
        self.stats = {"chunks_processed": 0, "total_confidence": 0.0, "start_time": datetime.now()}
    
    def setup_microphone(self):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.args.energy_threshold
        self.recorder.dynamic_energy_threshold = self.args.dynamic_energy
        self.recorder.pause_threshold = self.args.pause_threshold
        
        if sys.platform.startswith('linux') and self.args.default_microphone != 'default':
            self.setup_linux_microphone()
        else:
            self.source = sr.Microphone(sample_rate=SAMPLE_RATE)
            
        print("Calibrating microphone for ambient noise...")
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source, duration=1)
        print(f"Microphone calibrated. Energy threshold: {self.recorder.energy_threshold}")
    
    def setup_linux_microphone(self):
        mic_name = self.args.default_microphone
        if mic_name == 'list':
            self.list_microphones()
            sys.exit(0)
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                self.source = sr.Microphone(sample_rate=SAMPLE_RATE, device_index=index)
                return
        print(f"Microphone '{mic_name}' not found. Using default.")
        self.source = sr.Microphone(sample_rate=SAMPLE_RATE)
    
    def list_microphones(self):
        print("\nAvailable microphone devices:")
        print("-" * 40)
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{index:2d}: {name}")
    
    def record_callback(self, _, audio: sr.AudioData):
        self.data_queue.put(audio.get_raw_data())
    
    def display_interface(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("  ADVANCED REAL-TIME SPEECH TRANSCRIPTION & TTS")
        print("=" * 60)
        print(f"Model: {self.args.model} | Device: {self.transcriber.device} | Lang: Hinglish/Hindi")
        runtime = datetime.now() - self.stats["start_time"]
        avg_conf = (self.stats['total_confidence'] / self.stats['chunks_processed']) if self.stats['chunks_processed'] > 0 else 0
        print(f"  Runtime: {str(runtime).split('.')[0]} | Chunks: {self.stats['chunks_processed']} | Avg Conf: {avg_conf:.2f}")
        print("-" * 60)
        print(self.transcriber.buffer.get_display_text())
        print("\n" + "─" * 60)
        print("Press Ctrl+C to stop and save transcript")
    
    def run(self, tts_engine: Optional[AdvancedTTS] = None):
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.args.record_timeout)
        print("\n--- Listening... ---")
        while True:
            self.process_audio_queue(tts_engine)
            self.display_interface()
            time.sleep(0.1)
    
    def process_audio_queue(self, tts_engine: Optional[AdvancedTTS] = None):
        now = datetime.now()
        if not self.data_queue.empty():
            phrase_complete = False
            if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.args.phrase_timeout):
                self.last_sample = bytes()
                phrase_complete = True
            self.phrase_time = now

            audio_buffer = self.last_sample
            while not self.data_queue.empty():
                audio_buffer += self.data_queue.get()
            self.last_sample = audio_buffer

            audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            text, confidence = self.transcriber.transcribe_chunk(audio_np)

            if text and confidence > self.args.confidence_threshold:
                self.stats["chunks_processed"] += 1
                self.stats["total_confidence"] += confidence
                
                if phrase_complete:
                    lang = detect_language(text)
                    translation = translate_text(text, lang) if lang in ["hi", "mr", "ta"] else ""
                    self.transcriber.buffer.add_line(text, confidence, lang, translation)
                    if tts_engine and translation and lang != 'en':
                        tts_engine.speak(translation)
                else:
                    self.transcriber.buffer.update_current(text)
            elif not text:
                self.transcriber.buffer.update_current("")

    def cleanup(self):
        print("\nCleaning up and saving transcript...")
        output_dir = Path("transcripts")
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / f"transcript_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        self.transcriber.buffer.export_transcript(str(filename))
        print(f"Transcript saved to {filename}")
        print("Final transcript:")
        print(self.transcriber.buffer.get_display_text())

def main():
    # Initialize Text-to-Speech Engine
    print("\nInitializing Text-to-Speech engine...")
    tts_engine = None
    try:
        tts_config = TTSConfig(language='en', engine='edge', speed=1.1, volume=0.9)
        tts_engine = AdvancedTTS(config=tts_config)
        print("TTS engine initialized successfully.")
    except Exception as e:
        print(f"Could not initialize TTS engine: {e}")

    install_translation_models()

    parser = argparse.ArgumentParser(description="Advanced Real-Time Speech Transcription System", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model to use (default: base)")
    parser.add_argument("--device", default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to run model on (default: auto)")
    
    # Audio settings
    parser.add_argument("--energy_threshold", default=1000, type=int,
                       help="Energy level for mic to detect speech (default: 1000)")
    
    parser.add_argument("--record_timeout", default=2.0, type=float,
                       help="Recording chunk duration in seconds (default: 2.0)")
    
    parser.add_argument("--phrase_timeout", default=3.0, type=float,
                       help="Pause duration before considering new phrase (default: 3.0)")
    
    parser.add_argument("--confidence_threshold", default=0.5, type=float,
                       help="Minimum confidence score to display text (default: 0.5)")
    
    parser.add_argument("--dynamic_energy", action='store_true',
                       help="Enable dynamic energy threshold adjustment")
    
    # Microphone settings
    if sys.platform.startswith('linux'):
        parser.add_argument("--default_microphone", default='pulse', type=str,
                           help="Default microphone name for Linux (default: pulse)")
    
    parser.add_argument("--list_mics", action='store_true',
                       help="List available microphones and exit")
    
    args = parser.parse_args()
    
    # List microphones if requested
    if args.list_mics:
        transcriber = RealTimeTranscriber(args)
        transcriber.list_microphones()
        return
    
    # Validate arguments
    if not (0.0 <= args.confidence_threshold <= 1.0):
        parser.error("confidence_threshold must be between 0.0 and 1.0")
    
    print("🚀 Initializing Advanced Real-Time Transcription System...")
    print(f"📱 Model: {args.model}")
    print(f"🖥️  Device: {args.device}")
    print(f"🎯 Confidence Threshold: {args.confidence_threshold}")
    
    # Create and run transcriber
    transcriber = RealTimeTranscriber(args)
    try:
        transcriber.run(tts_engine=tts_engine)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting.")
    finally:
        if tts_engine:
            print("\nShutting down TTS engine...")
            tts_engine.cleanup()
        transcriber.cleanup()

if __name__ == "__main__":
    main()
