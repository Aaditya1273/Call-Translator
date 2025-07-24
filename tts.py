#!/usr/bin/env python3
"""
Advanced Real-Time Text-to-Speech System
Ultra-fast, dynamic, and smooth TTS with multi-language support
"""

import asyncio
import threading
import queue
import os
import sys
import json
import time
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Audio and TTS imports
import pygame
import pyttsx3
from gtts import gTTS
import edge_tts
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import io
import wave

# Suppress warnings
warnings.filterwarnings("ignore")
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

@dataclass
class TTSConfig:
    """Configuration for TTS system"""
    language: str = 'en'
    voice_gender: str = 'female'  # 'male', 'female', 'auto'
    speed: float = 1.0  # 0.5 to 2.0
    volume: float = 0.8  # 0.0 to 1.0
    engine: str = 'edge'  # 'gtts', 'pyttsx3', 'edge', 'auto'
    cache_enabled: bool = True
    streaming: bool = True
    max_cache_size: int = 100  # MB
    chunk_size: int = 1000  # characters per chunk for long texts

class AudioCache:
    """Intelligent audio caching system"""
    
    def __init__(self, max_size_mb: int = 100):
        self.cache_dir = Path(tempfile.gettempdir()) / "advanced_tts_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict:
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f)
    
    def _get_cache_key(self, text: str, config: TTSConfig) -> str:
        """Generate cache key for text and config"""
        cache_string = f"{text}_{config.language}_{config.engine}_{config.speed}_{config.voice_gender}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, text: str, config: TTSConfig) -> Optional[str]:
        """Get cached audio file path"""
        key = self._get_cache_key(text, config)
        if key in self.cache_index:
            file_path = self.cache_dir / f"{key}.mp3"
            if file_path.exists():
                # Update access time
                self.cache_index[key]['last_accessed'] = time.time()
                return str(file_path)
        return None
    
    def set(self, text: str, config: TTSConfig, audio_file: str):
        """Cache audio file"""
        key = self._get_cache_key(text, config)
        cached_file = self.cache_dir / f"{key}.mp3"
        
        # Copy file to cache
        if os.path.exists(audio_file):
            import shutil
            shutil.copy2(audio_file, cached_file)
            
            # Update index
            self.cache_index[key] = {
                'text': text[:100],  # Store first 100 chars for reference
                'created': time.time(),
                'last_accessed': time.time(),
                'size': os.path.getsize(cached_file)
            }
            
            self._cleanup_cache()
            self._save_cache_index()
    
    def _cleanup_cache(self):
        """Remove old cache entries if size limit exceeded"""
        total_size = sum(entry['size'] for entry in self.cache_index.values())
        
        if total_size > self.max_size_bytes:
            # Sort by last accessed time
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            # Remove oldest entries
            for key, entry in sorted_entries:
                if total_size <= self.max_size_bytes * 0.8:  # Keep 80% of limit
                    break
                    
                file_path = self.cache_dir / f"{key}.mp3"
                if file_path.exists():
                    file_path.unlink()
                    total_size -= entry['size']
                    del self.cache_index[key]

class VoiceEngine:
    """Base class for TTS engines"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
    
    async def synthesize(self, text: str) -> str:
        """Synthesize text to audio file"""
        raise NotImplementedError
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        return []

class EdgeTTSEngine(VoiceEngine):
    """Microsoft Edge TTS Engine - Fastest and highest quality"""
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.voices = {
            'en': {
                'female': 'en-US-AriaNeural',
                'male': 'en-US-GuyNeural'
            },
            'hi': {
                'female': 'hi-IN-SwaraNeural', 
                'male': 'hi-IN-MadhurNeural'
            },
            'es': {
                'female': 'es-ES-ElviraNeural',
                'male': 'es-ES-AlvaroNeural'
            },
            'fr': {
                'female': 'fr-FR-DeniseNeural',
                'male': 'fr-FR-HenriNeural'
            },
            'de': {
                'female': 'de-DE-KatjaNeural',
                'male': 'de-DE-ConradNeural'
            }
        }
    
    def _get_voice(self) -> str:
        """Get appropriate voice for language and gender"""
        lang_voices = self.voices.get(self.config.language, self.voices['en'])
        return lang_voices.get(self.config.voice_gender, list(lang_voices.values())[0])
    
    async def synthesize(self, text: str) -> str:
        """Synthesize using Edge TTS"""
        voice = self._get_voice()
        
        # Create rate string for speed control
        rate_percent = int((self.config.speed - 1.0) * 100)
        rate = f"{rate_percent:+d}%" if rate_percent != 0 else "+0%"
        
        # Create volume string
        volume_percent = int(self.config.volume * 100)
        volume = f"{volume_percent}%"
        
        # Generate SSML for better control
        ssml = f"""
        <speak version=\"1.0\" xmlns=\"http://www.w3.org/2001/10/synthesis\" xml:lang=\"{self.config.language}\">
            <voice name=\"{voice}\">
                <prosody rate=\"{rate}\" volume=\"{volume}\">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()
        
        try:
            communicate = edge_tts.Communicate(ssml, voice)
            await communicate.save(temp_file.name)
            return temp_file.name
        except Exception as e:
            print(f"Edge TTS error: {e}")
            # Fallback to simple text
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(temp_file.name)
            return temp_file.name

class GTTSEngine(VoiceEngine):
    """Google TTS Engine - Good fallback option"""
    
    async def synthesize(self, text: str) -> str:
        """Synthesize using gTTS"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()
        
        try:
            # Adjust speed by changing the slow parameter and post-processing
            slow = self.config.speed < 0.8
            tts = gTTS(text=text, lang=self.config.language, slow=slow)
            tts.save(temp_file.name)
            
            # Post-process for speed and volume if needed
            if self.config.speed != 1.0 or self.config.volume != 1.0:
                audio = AudioSegment.from_mp3(temp_file.name)
                
                # Adjust speed
                if self.config.speed != 1.0:
                    # Change frame rate for speed adjustment
                    audio = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate * self.config.speed)
                    }).set_frame_rate(audio.frame_rate)
                
                # Adjust volume
                if self.config.volume != 1.0:
                    volume_change = 20 * np.log10(self.config.volume)  # Convert to dB
                    audio = audio + volume_change
                
                audio.export(temp_file.name, format="mp3")
            
            return temp_file.name
            
        except Exception as e:
            print(f"gTTS error: {e}")
            raise

class PyttsxEngine(VoiceEngine):
    """Pyttsx3 Engine - Offline option"""
    
    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.engine = pyttsx3.init()
        self._configure_engine()
    
    def _configure_engine(self):
        """Configure pyttsx3 engine"""
        voices = self.engine.getProperty('voices')
        
        # Set voice based on gender preference
        if voices:
            if self.config.voice_gender == 'female':
                # Try to find female voice
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            elif self.config.voice_gender == 'male':
                # Try to find male voice
                for voice in voices:
                    if 'male' in voice.name.lower() or 'david' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
        
        # Set rate and volume
        self.engine.setProperty('rate', int(200 * self.config.speed))
        self.engine.setProperty('volume', self.config.volume)
    
    async def synthesize(self, text: str) -> str:
        """Synthesize using pyttsx3"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()
        
        try:
            self.engine.save_to_file(text, temp_file.name)
            self.engine.runAndWait()
            
            # Convert WAV to MP3 for consistency
            audio = AudioSegment.from_wav(temp_file.name)
            mp3_file = temp_file.name.replace('.wav', '.mp3')
            audio.export(mp3_file, format="mp3")
            
            # Clean up WAV file
            os.unlink(temp_file.name)
            
            return mp3_file
            
        except Exception as e:
            print(f"Pyttsx3 error: {e}")
            raise

class AdvancedTTS:
    """Advanced Text-to-Speech System with real-time capabilities"""
    
    def __init__(self, config: TTSConfig = None):
        self.config = config or TTSConfig()
        self.cache = AudioCache(self.config.max_cache_size) if self.config.cache_enabled else None
        self.engines = self._initialize_engines()
        self.current_engine = self._select_best_engine()
        
        # Threading and queuing for real-time processing
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'synthesis_time': 0,
            'playback_time': 0
        }
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
    
    def _initialize_engines(self) -> Dict[str, VoiceEngine]:
        """Initialize all available TTS engines"""
        engines = {}
        
        try:
            engines['edge'] = EdgeTTSEngine(self.config)
        except:
            pass
            
        try:
            engines['gtts'] = GTTSEngine(self.config)
        except:
            pass
            
        try:
            engines['pyttsx3'] = PyttsxEngine(self.config)
        except:
            pass
            
        return engines
    
    def _select_best_engine(self) -> VoiceEngine:
        """Select the best available engine"""
        priority = ['edge', 'gtts', 'pyttsx3']
        
        if self.config.engine != 'auto' and self.config.engine in self.engines:
            return self.engines[self.config.engine]
        
        for engine_name in priority:
            if engine_name in self.engines:
                return self.engines[engine_name]
        
        raise RuntimeError("No TTS engines available")
    
    def _speech_worker(self):
        """Background worker for processing speech queue"""
        while True:
            try:
                task = self.speech_queue.get(timeout=1)
                if task is None:  # Shutdown signal
                    break
                    
                text, callback, priority = task
                self._process_speech_task(text, callback)
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speech worker error: {e}")
    
    def _process_speech_task(self, text: str, callback: Optional[Callable] = None):
        """Process individual speech task"""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_file = None
            if self.cache:
                cached_file = self.cache.get(text, self.config)
                if cached_file:
                    self.stats['cache_hits'] += 1
            
            if cached_file:
                audio_file = cached_file
            else:
                # Synthesize new audio
                synthesis_start = time.time()
                
                if asyncio.iscoroutinefunction(self.current_engine.synthesize):
                    # Run async synthesis in thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    audio_file = loop.run_until_complete(self.current_engine.synthesize(text))
                    loop.close()
                else:
                    audio_file = self.current_engine.synthesize(text)
                
                self.stats['synthesis_time'] += time.time() - synthesis_start
                
                # Cache the result
                if self.cache and audio_file:
                    self.cache.set(text, self.config, audio_file)
            
            # Play audio
            if audio_file and os.path.exists(audio_file):
                playback_start = time.time()
                self._play_audio_optimized(audio_file, not cached_file)
                self.stats['playback_time'] += time.time() - playback_start
            
            self.stats['total_requests'] += 1
            
            if callback:
                callback(True, time.time() - start_time)
                
        except Exception as e:
            print(f"Error processing speech: {e}")
            if callback:
                callback(False, time.time() - start_time)
    
    def _play_audio_optimized(self, audio_file: str, cleanup: bool = True):
        """Optimized audio playback using pygame"""
        try:
            # Load and play with pygame for better performance
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except Exception as e:
            print(f"Pygame playback failed, trying fallback: {e}")
            try:
                # Fallback to pydub
                audio = AudioSegment.from_file(audio_file)
                play(audio)
            except Exception as e2:
                print(f"Audio playback failed: {e2}")
        
        finally:
            # Cleanup temporary file
            if cleanup and audio_file and os.path.exists(audio_file):
                try:
                    os.unlink(audio_file)
                except:
                    pass
    
    def speak(self, text: str, priority: int = 1, callback: Optional[Callable] = None) -> bool:
        """
        Add text to speech queue
        
        Args:
            text: Text to speak
            priority: Priority level (higher = more important)
            callback: Optional callback function(success: bool, duration: float)
        
        Returns:
            bool: True if queued successfully
        """
        if not text or not text.strip():
            if callback:
                callback(False, 0)
            return False
        
        try:
            self.speech_queue.put((text.strip(), callback, priority))
            return True
        except Exception as e:
            print(f"Error queuing speech: {e}")
            if callback:
                callback(False, 0)
            return False
    
    def speak_now(self, text: str) -> bool:
        """
        Speak text immediately (blocking)
        
        Args:
            text: Text to speak
            
        Returns:
            bool: True if successful
        """
        if not text or not text.strip():
            return False
        
        result = {'success': False}
        
        def callback(success, duration):
            result['success'] = success
        
        self.speak(text, priority=10, callback=callback)
        
        # Wait for completion
        self.speech_queue.join()
        return result['success']
    
    def speak_streaming(self, text: str, chunk_size: int = None) -> bool:
        """
        Stream long text by breaking into chunks
        
        Args:
            text: Long text to speak
            chunk_size: Size of each chunk (uses config default if None)
            
        Returns:
            bool: True if queued successfully
        """
        if not text or not text.strip():
            return False
        
        chunk_size = chunk_size or self.config.chunk_size
        
        # Split text into sentences for better chunking
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Queue all chunks
        success = True
        for i, chunk in enumerate(chunks):
            if not self.speak(chunk, priority=5-i):  # Decreasing priority
                success = False
        
        return success
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def stop(self):
        """Stop current speech and clear queue"""
        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
        
        # Stop pygame mixer
        try:
            pygame.mixer.music.stop()
        except:
            pass
    
    def wait_until_done(self, timeout: float = None):
        """Wait until all queued speech is completed"""
        self.speech_queue.join()
    
    def is_busy(self) -> bool:
        """Check if TTS is currently processing"""
        return not self.speech_queue.empty() or pygame.mixer.music.get_busy()
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        stats = self.stats.copy()
        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
            stats['avg_synthesis_time'] = stats['synthesis_time'] / (stats['total_requests'] - stats['cache_hits']) if stats['total_requests'] > stats['cache_hits'] else 0
            stats['avg_playback_time'] = stats['playback_time'] / stats['total_requests']
        return stats
    
    def set_config(self, **kwargs):
        """Update configuration dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize engines if needed
        if any(key in ['engine', 'language', 'voice_gender'] for key in kwargs):
            self.engines = self._initialize_engines()
            self.current_engine = self._select_best_engine()
    
    def list_voices(self) -> Dict[str, List[str]]:
        """List available voices for each engine"""
        voices = {}
        for name, engine in self.engines.items():
            try:
                voices[name] = engine.get_available_voices()
            except:
                voices[name] = []
        return voices
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        self.speech_queue.put(None)  # Shutdown signal
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1)
        self.executor.shutdown(wait=True)

# Convenience functions for backward compatibility
def speak_english(text: str, **kwargs):
    """Simple English TTS function"""
    config = TTSConfig(language='en', **kwargs)
    tts = AdvancedTTS(config)
    return tts.speak_now(text)

def speak_hindi(text: str, **kwargs):
    """Simple Hindi TTS function"""
    config = TTSConfig(language='hi', **kwargs)
    tts = AdvancedTTS(config)
    return tts.speak_now(text)

def speak_multilingual(text: str, language: str = 'en', **kwargs):
    """Multi-language TTS function"""
    config = TTSConfig(language=language, **kwargs)
    tts = AdvancedTTS(config)
    return tts.speak_now(text)

# Demo and testing functions
async def demo_advanced_features():
    """Demonstrate advanced TTS features"""
    print("🎤 Advanced TTS System Demo")
    print("=" * 50)
    
    # Create TTS with custom config
    config = TTSConfig(
        language='en',
        voice_gender='female',
        speed=1.2,
        volume=0.9,
        engine='edge',
        streaming=True
    )
    
    tts = AdvancedTTS(config)
    
    print("1. Basic speech...")
    tts.speak_now("Hello! This is the advanced text-to-speech system.")
    
    print("2. Queue multiple texts...")
    tts.speak("First message in queue.")
    tts.speak("Second message in queue.")
    tts.speak("Third message in queue.")
    tts.wait_until_done()
    
    print("3. Streaming long text...")
    long_text = """
    This is a demonstration of the streaming capability. 
    The system can break down long texts into smaller chunks 
    and speak them sequentially for better performance and user experience.
    Each chunk is processed and cached independently for maximum efficiency.
    """
    tts.speak_streaming(long_text)
    tts.wait_until_done()
    
    print("4. Multi-language demo...")
    # Switch to Hindi
    tts.set_config(language='hi', voice_gender='female')
    tts.speak_now("नमस्ते! यह हिंदी में बोल रहा है।")
    
    # Switch back to English
    tts.set_config(language='en', voice_gender='male', speed=0.9)
    tts.speak_now("Now speaking in English with male voice at slower speed.")
    
    print("5. Performance statistics:")
    stats = tts.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    tts.cleanup()
    print("Demo completed!")

if __name__ == "__main__":
    # Simple usage examples
    print("🚀 Advanced TTS System - Quick Test")
    
    # Test 1: Simple English
    print("Test 1: Basic English TTS")
    speak_english("Hello, this is a test of the advanced text to speech system!")
    
    # Test 2: Hindi
    print("Test 2: Hindi TTS")
    speak_hindi("नमस्ते! यह एक उन्नत टेक्स्ट टू स्पीच सिस्टम है।")
    
    # Test 3: Advanced usage
    print("Test 3: Advanced TTS with custom settings")
    config = TTSConfig(
        language='en',
        voice_gender='female',
        speed=1.3,
        volume=0.8,
        engine='edge'
    )
    
    advanced_tts = AdvancedTTS(config)
    
    # Demonstrate async capabilities
    texts = [
        "This is the first message.",
        "Here comes the second message.",
        "And finally, the third message with advanced processing!"
    ]
    
    for text in texts:
        advanced_tts.speak(text)
    
    advanced_tts.wait_until_done()
    
    print("✅ All tests completed!")
    print(f"📊 Performance stats: {advanced_tts.get_stats()}")
    
    advanced_tts.cleanup()

