import streamlit as st
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import io
import pandas as pd
import os
from datetime import datetime

from core import (
    SmartTranscriber,
    detect_language,
    translate_text,
    get_tts_language_code,
    get_optimal_device
)
from tts import AdvancedTTS, TTSConfig

st.set_page_config(layout="wide")

st.title("🎤 Bidirectional Speech Translator 💬")
st.markdown("--- ")

# --- STATE MANAGEMENT ---
def initialize_state():
    if 'mode' not in st.session_state:
        st.session_state.mode = "Real-time Translation"
        st.session_state.transcriber = None
        st.session_state.original_text = ""
        st.session_state.detected_lang = ""
        st.session_state.translated_text = ""
        st.session_state.audio_output = None
        st.session_state.chat_history = []
        st.session_state.current_speaker = "A"
        st.session_state.speaker_a_lang = "hi"
        st.session_state.speaker_b_lang = "en"

initialize_state()

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    device = get_optimal_device("auto")
    transcriber = SmartTranscriber(model_name="base", device=device)
    tts_engine = AdvancedTTS(config=TTSConfig())
    return transcriber, tts_engine

with st.spinner("Loading models, please wait..."):
    transcriber, tts_engine = load_models()

st.session_state.transcriber = transcriber

LOG_FILE = "conversation_log.csv"

# --- LOGGING FUNCTION ---
def log_to_csv(data):
    df = pd.DataFrame([data])
    file_exists = os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Settings")
st.session_state.mode = st.sidebar.radio("Choose Mode", ["Real-time Translation", "Two-Person Chat"])

LANGUAGE_OPTIONS = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta"}

if st.session_state.mode == "Two-Person Chat":
    st.sidebar.subheader("Speaker Configuration")
    speaker_a_lang_name = st.sidebar.selectbox("Speaker A Language", list(LANGUAGE_OPTIONS.keys()), index=1)
    speaker_b_lang_name = st.sidebar.selectbox("Speaker B Language", list(LANGUAGE_OPTIONS.keys()), index=0)
    st.session_state.speaker_a_lang = LANGUAGE_OPTIONS[speaker_a_lang_name]
    st.session_state.speaker_b_lang = LANGUAGE_OPTIONS[speaker_b_lang_name]

    st.sidebar.subheader("Turn Management")
    if st.sidebar.button(f"Switch to Speaker {'B' if st.session_state.current_speaker == 'A' else 'A'}"):
        st.session_state.current_speaker = 'B' if st.session_state.current_speaker == 'A' else 'A'
    st.sidebar.info(f"**Current Speaker: {st.session_state.current_speaker}** ({speaker_a_lang_name if st.session_state.current_speaker == 'A' else speaker_b_lang_name}) ")

# --- AUDIO PROCESSING ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.array([], dtype=np.int16)
        self.last_process_time = time.time()

    def recv(self, frame):
        audio_data = frame.to_ndarray(format="s16", layout="mono")
        self.buffer = np.append(self.buffer, audio_data)
        current_time = time.time()
        if current_time - self.last_process_time > 3.0: # Process every 3 seconds
            self.process_audio()
            self.last_process_time = current_time
        return frame

    def process_audio(self):
        if len(self.buffer) == 0 or not st.session_state.transcriber:
            return

        audio_np = self.buffer.astype(np.float32) / 32768.0
        self.buffer = np.array([], dtype=np.int16)
        text, confidence = st.session_state.transcriber.transcribe_chunk(audio_np)

        if not (text and confidence > 0.4):
            return

        if st.session_state.mode == "Real-time Translation":
            self.handle_realtime_translation(text)
        else:
            self.handle_chat_translation(text)

    def handle_realtime_translation(self, text):
        st.session_state.original_text = text
        lang = detect_language(text)
        st.session_state.detected_lang = lang.upper()
        
        target_lang = "en" if lang != "en" else st.session_state.speaker_a_lang
        translation = translate_text(text, lang, target_lang)
        st.session_state.translated_text = translation
        
        if translation:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "speaker": "N/A (Real-time)",
                "original_text": text,
                "source_language": lang,
                "translated_text": translation,
                "target_language": target_lang
            }
            log_to_csv(log_entry)
            self.generate_tts(translation, target_lang)

    def handle_chat_translation(self, text):
        speaker = st.session_state.current_speaker
        source_lang = st.session_state.speaker_a_lang if speaker == 'A' else st.session_state.speaker_b_lang
        target_lang = st.session_state.speaker_b_lang if speaker == 'A' else st.session_state.speaker_a_lang

        translation = translate_text(text, source_lang, target_lang)
        
        chat_entry = {
            "speaker": speaker,
            "original": text,
            "translated": translation,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        st.session_state.chat_history.append(chat_entry)
        
        if translation:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "speaker": speaker,
                "original_text": text,
                "source_language": source_lang,
                "translated_text": translation,
                "target_language": target_lang
            }
            log_to_csv(log_entry)
            self.generate_tts(translation, target_lang)

    def generate_tts(self, text, lang):
        try:
            tts_lang_code = get_tts_language_code(lang)
            audio_bytes = tts_engine.say(text, language=tts_lang_code)
            st.session_state.audio_output = audio_bytes
        except Exception as e:
            st.error(f"TTS Error: {e}")
            st.session_state.audio_output = None

# --- UI DISPLAY ---
tab1, tab2 = st.tabs(["Translator", "Conversation Log"])

with tab1:
    webrtc_streamer(
        key="speech-translator",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if st.session_state.mode == "Real-time Translation":
        col1, col2 = st.columns(2)
        with col1:
            st.header("Live Transcription")
            st.info(f"**Original:** {st.session_state.original_text}")
            st.write(f"**Detected Language:** {st.session_state.detected_lang}")
        with col2:
            st.header("Translation")
            st.success(f"**Translation:** {st.session_state.translated_text}")
            if st.session_state.audio_output:
                st.audio(st.session_state.audio_output, format="audio/wav")
                st.session_state.audio_output = None
    else: # Two-Person Chat Mode
        st.header("Conversation Log")
        for entry in reversed(st.session_state.chat_history):
            with st.chat_message("user" if entry["speaker"] == 'A' else "assistant"):
                st.write(f"**Speaker {entry['speaker']} ({entry['source_lang']}):** {entry['original']}")
                st.write(f"**Translation ({entry['target_lang']}):** {entry['translated']}")

        if st.session_state.audio_output:
            st.audio(st.session_state.audio_output, format="audio/wav", autoplay=True)
            st.session_state.audio_output = None

with tab2:
    st.header("Full Conversation History")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df)
        if st.button("Clear Log"):
            os.remove(LOG_FILE)
            st.rerun()
    else:
        st.info("No conversation history found.")
