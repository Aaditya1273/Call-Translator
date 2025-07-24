#!/usr/bin/env python3
"""
Command-line interface for the Real-Time Speech Transcription System.
"""

import os
import sys
import argparse
import time
from queue import Queue
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import speech_recognition as sr

from core import (
    SmartTranscriber, 
    detect_language, 
    translate_text, 
    get_target_language_for_english, 
    get_tts_language_code,
    install_translation_models,
    get_optimal_device,
    SAMPLE_RATE
)
from tts import AdvancedTTS, TTSConfig

class RealTimeTranscriber:
    """Main real-time transcription system for the CLI."""
    
    def __init__(self, args):
        self.args = args
        self.transcriber = SmartTranscriber(args.model, args.device)
        self.data_queue = Queue()
        self.phrase_time = None
        self.last_sample = bytes()
        self.setup_microphone()
        self.stats = {"chunks_processed": 0, "total_confidence": 0.0, "start_time": datetime.now()}

    def setup_microphone(self):
        mic_name = self.args.default_microphone
        if mic_name == "default":
            self.source = sr.Microphone(sample_rate=SAMPLE_RATE)
        else:
            mic_list = sr.Microphone.list_microphone_names()
            for i, m in enumerate(mic_list):
                if mic_name in m:
                    self.source = sr.Microphone(device_index=i, sample_rate=SAMPLE_RATE)
                    break
            else:
                print(f"Microphone '{mic_name}' not found. Using default.")
                self.source = sr.Microphone(sample_rate=SAMPLE_RATE)

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.args.energy_threshold
        self.recorder.dynamic_energy_threshold = self.args.dynamic_energy
        self.recorder.pause_threshold = self.args.pause_threshold

        print("\nCalibrating microphone... Please wait.")
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source, duration=2)
        print(f"Microphone calibrated with energy threshold: {self.recorder.energy_threshold:.2f}")
    
    @staticmethod
    def list_microphones():
        print("\nAvailable microphone devices:")
        print("-" * 40)
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"  {index}: {name}")
        print("-" * 40)
        print("Use --default_microphone 'name' to select a microphone.")
    
    def record_callback(self, _, audio: sr.AudioData):
        self.data_queue.put(audio.get_raw_data())
    
    def display_interface(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("  BIDIRECTIONAL REAL-TIME TRANSCRIPTION & TTS (CLI)")
        print("="*50)
        print(f"  Model: {self.args.model} | Device: {self.args.device} | Target Lang: {self.args.target_language.upper()}")
        print("-"*50)
        print(self.transcriber.buffer.get_display_text())
        print("\n" + "-"*50)
        
        now = datetime.now()
        elapsed_time = (now - self.stats['start_time']).total_seconds()
        avg_confidence = (self.stats['total_confidence'] / self.stats['chunks_processed']) if self.stats['chunks_processed'] > 0 else 0
        
        print(f"  Status: Listening... | Chunks: {self.stats['chunks_processed']} | Confidence: {avg_confidence:.2f} | Uptime: {timedelta(seconds=int(elapsed_time))}")
        print("="*50)
        print("  Say 'quit' or 'exit' to stop.")

    def run(self, tts_engine: Optional[AdvancedTTS] = None):
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.args.record_timeout)
        print("\nSystem is running. Press Ctrl+C to exit.")
        self.process_audio_queue(tts_engine)

    def process_audio_queue(self, tts_engine: Optional[AdvancedTTS] = None):
        while True:
            try:
                now = datetime.now()
                if not self.data_queue.empty():
                    self.phrase_time = now + timedelta(seconds=self.args.phrase_timeout)
                    
                    audio_data = b''.join(self.data_queue.queue)
                    self.data_queue.queue.clear()
                    
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.last_sample = audio_data

                    text, confidence = self.transcriber.transcribe_chunk(audio_np)
                    if text and confidence > self.args.confidence_threshold:
                        self.stats["chunks_processed"] += 1
                        self.stats["total_confidence"] += confidence
                        self.transcriber.buffer.update_current(text)
                        
                        if any(word in text.lower() for word in ["quit", "exit"]):
                            break

                elif self.phrase_time and now > self.phrase_time:
                    if self.last_sample:
                        text = self.transcriber.buffer.current_line
                        confidence = self.stats['total_confidence'] / self.stats['chunks_processed'] if self.stats['chunks_processed'] > 0 else 0.5
                        
                        lang = detect_language(text)
                        translation = ""
                        target_lang_for_tts = "en-US"
                        
                        if lang in ["hi", "mr", "ta"]:
                            translation = translate_text(text, lang, "en")
                            target_lang_for_tts = get_tts_language_code("en")
                        elif lang == "en":
                            target_lang = get_target_language_for_english(self.args.target_language)
                            translation = translate_text(text, "en", target_lang)
                            target_lang_for_tts = get_tts_language_code(target_lang)
                        
                        self.transcriber.buffer.add_line(text, confidence, lang, translation, self.args.target_language)
                        
                        if tts_engine and translation:
                            tts_engine.say(translation, language=target_lang_for_tts)

                    self.phrase_time = None
                    self.last_sample = bytes()
                
                self.display_interface()
                time.sleep(0.25)
            except Exception as e:
                print(f"Error in processing loop: {e}")
                break

    def cleanup(self):
        print("\nCleaning up and exporting transcript...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.txt"
        self.transcriber.buffer.export_transcript(filename)
        print(f"Transcript saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Command-line interface for the speech translator.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model to use.")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "mps", "auto"], help="Device for computation.")
    parser.add_argument("--energy_threshold", default=1000, type=int, help="Energy level for mic to detect speech.")
    parser.add_argument("--dynamic_energy", action='store_true', help="Enable dynamic energy threshold.")
    parser.add_argument("--record_timeout", default=2.0, type=float, help="How long to record before processing.")
    parser.add_argument("--phrase_timeout", default=3.0, type=float, help="How long to wait for a phrase to end.")
    parser.add_argument("--pause_threshold", default=0.8, type=float, help="Seconds of non-speaking audio before a phrase is considered complete.")
    parser.add_argument("--confidence_threshold", default=0.3, type=float, help="Confidence threshold for accepting transcription.")
    parser.add_argument("--default_microphone", default='default', help='Name of the microphone to use.')
    parser.add_argument("--target_language", default="hi", choices=["hi", "ta"], help="Target language for English input.")
    parser.add_argument("--list_mics", action='store_true', help="List available microphones and exit.")
    args = parser.parse_args()

    if args.list_mics:
        RealTimeTranscriber.list_microphones()
        sys.exit(0)

    args.device = get_optimal_device(args.device)

    print("\nInitializing TTS engine...")
    tts_engine = None
    try:
        tts_config = TTSConfig(language='en', engine='edge', speed=1.1, volume=0.9)
        tts_engine = AdvancedTTS(config=tts_config)
        print("TTS engine initialized.")
    except Exception as e:
        print(f"Could not initialize TTS engine: {e}")

    install_translation_models()

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
