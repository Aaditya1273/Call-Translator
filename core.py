import os
import sys
from typing import Tuple
from collections import deque
from datetime import datetime

import argostranslate.package
import argostranslate.translate
import torch
import whisper
import numpy as np
from scipy import signal
import webrtcvad
from langdetect import detect, DetectorFactory

# Constants
SAMPLE_RATE = 16000

# Ensure consistent language detection results
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
}

def install_translation_models():
    """Download and install Argos Translate models for supported languages (bidirectional)."""
    print("\nChecking and installing translation models...")
    available_packages = argostranslate.package.get_available_packages()
    installed_languages = [lang.code for lang in argostranslate.translate.get_installed_languages()]

    for from_code in SUPPORTED_LANGUAGES:
        for to_code in SUPPORTED_LANGUAGES:
            if from_code == to_code:
                continue

            # Check if model is already installed
            if from_code in installed_languages and to_code in installed_languages:
                try:
                    argostranslate.translate.translate("", from_code, to_code)
                    continue
                except Exception:
                    pass

            # Find and install the package
            package_to_install = next(
                (p for p in available_packages if p.from_code == from_code and p.to_code == to_code), None
            )
            if package_to_install:
                print(f"  Downloading and installing model: {from_code} -> {to_code}")
                argostranslate.package.install_from_path(package_to_install.download())
            else:
                print(f"  Warning: Translation model not found for {from_code} -> {to_code}")
    print("Translation models are up to date.")

def detect_language(text: str) -> str:
    """Detect the language of a given text."""
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "unknown"

def translate_text(text: str, from_code: str, to_code: str = "en") -> str:
    """Translate text from a source language to a target language (bidirectional)."""
    try:
        return argostranslate.translate.translate(text, from_code, to_code)
    except Exception as e:
        return text

def get_target_language_for_english(target_lang: str = "hi") -> str:
    """Get the target language code for English input translation."""
    return target_lang if target_lang in ["hi", "ta"] else "hi"

def get_tts_language_code(lang_code: str) -> str:
    """Map language codes to TTS-compatible language codes."""
    tts_map = {
        "en": "en-US",
        "hi": "hi-IN",
        "ta": "ta-IN",
        "mr": "mr-IN",
    }
    return tts_map.get(lang_code, "en-US")

def get_optimal_device(device: str) -> str:
    """Determine the optimal torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device

class AudioProcessor:
    """Advanced audio processing for better transcription quality"""
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(2)

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

    def add_line(self, text: str, confidence: float, lang: str, translation: str, target_lang: str = "en"):
        if text.strip():
            self.transcript.append((datetime.now(), text.strip(), lang, translation, confidence, target_lang))
            self.current_line = ""

    def update_current(self, text: str):
        self.current_line = text.strip()

    def get_display_text(self) -> str:
        display_lines = []
        for item in self.transcript:
            if len(item) == 6:
                _, line, lang, translation, _, target_lang = item
            else:
                _, line, lang, translation, _ = item
                target_lang = "en"
            display_line = f"[{lang.upper()}] {line}"
            if translation:
                target_display = target_lang.upper()
                display_line += f" → ({target_display}: {translation})"
            display_lines.append(display_line)
        if self.current_line:
            display_lines.append(f" {self.current_line}")
        return "\n".join(display_lines)

    def export_transcript(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Bidirectional Real-Time Transcription Log\n")
            f.write("="*40 + "\n")
            for item in self.transcript:
                if len(item) == 6:
                    timestamp, line, lang, translation, confidence, target_lang = item
                else:
                    timestamp, line, lang, translation, confidence = item
                    target_lang = "en"
                f.write(f"[{timestamp.strftime('%H:%M:%S')}] [{lang.upper()}] (Conf: {confidence:.2f}) {line}")
                if translation:
                    f.write(f" → ({target_lang.upper()}: {translation})")
                f.write("\n")

class SmartTranscriber:
    """Advanced transcription engine with Whisper"""
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model = self._load_optimized_model(model_name)
        self.audio_processor = AudioProcessor()
        self.buffer = TranscriptionBuffer()

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
                initial_prompt="This is a conversation in Hinglish."
            )
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            if not segments:
                return "", 0.0
            no_speech_prob = np.mean([seg.get("no_speech_prob", 1.0) for seg in segments])
            confidence = 1 - no_speech_prob
            return text, confidence
        except Exception as e:
            print(f"Transcription error: {e}")
            return "", 0.0
