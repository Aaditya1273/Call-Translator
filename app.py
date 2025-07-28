import streamlit as st
import queue
import threading
import time
from typing import List

import speech_recognition as sr
import numpy as np

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

st.set_page_config(layout="wide")

# --- State Management ---
def init_session_state():
    """Initializes all session state variables to prevent KeyErrors."""
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = None
    if 'audio_queue' not in st.session_state:
        st.session_state.audio_queue = queue.Queue()
    if 'run_transcription' not in st.session_state:
        st.session_state.run_transcription = False
    if 'transcript' not in st.session_state:
        st.session_state.transcript = []
    if 'tts_engine' not in st.session_state:
        st.session_state.tts_engine = None
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = sr.Recognizer()
    if 'microphone' not in st.session_state:
        st.session_state.microphone = sr.Microphone(sample_rate=SAMPLE_RATE)
    if 'target_language' not in st.session_state:
        st.session_state.target_language = "hi"

init_session_state()

def record_callback(_, audio: sr.AudioData):
    """Callback function to receive audio data from the microphone."""
    if st.session_state.run_transcription:
        st.session_state.audio_queue.put(audio.get_raw_data())

def process_audio():
    """Main audio processing loop."""
    while st.session_state.run_transcription:
        try:
            if st.session_state.transcriber is None:
                time.sleep(0.1)
                continue

            audio_data = st.session_state.audio_queue.get(timeout=1)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            text, confidence = st.session_state.transcriber.transcribe_chunk(audio_np)

            if text:
                detected_lang = detect_language(text)
                translation = ""
                
                if detected_lang == 'en':
                    target_lang = st.session_state.target_language
                    translation = translate_text(text, from_code='en', to_code=target_lang)
                    if st.session_state.tts_engine:
                        tts_lang_code = get_tts_language_code(target_lang)
                        st.session_state.tts_engine.say(translation, tts_lang_code)
                else:
                    translation = translate_text(text, from_code=detected_lang, to_code='en')
                
                st.session_state.transcript.append((detected_lang.upper(), text, translation))

        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"Error in processing thread: {e}")
            time.sleep(1)

# --- UI Layout ---
st.title("Real-Time Bidirectional Voice Translator")

with st.sidebar:
    st.header("Configuration")
    model_name = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"], index=1)
    target_language = st.selectbox("Translate English to", ["hi", "ta"], index=0)
    
    use_tts = st.checkbox("Enable Voice Output (TTS)", value=True)

    if st.button("Start Transcription"):
        if not st.session_state.run_transcription:
            st.session_state.run_transcription = True
            st.session_state.target_language = target_language
            device = get_optimal_device("auto")
            st.session_state.transcriber = SmartTranscriber(model_name, device)
            if use_tts:
                tts_config = TTSConfig()
                st.session_state.tts_engine = AdvancedTTS(config=tts_config)
            
            # Start listening in the background
            st.session_state.recognizer.listen_in_background(st.session_state.microphone, record_callback, phrase_time_limit=5)
            
            # Start the processing thread
            threading.Thread(target=process_audio, daemon=True).start()
            st.success("Transcription started!")
            st.rerun()

    if st.button("Stop Transcription"):
        if st.session_state.run_transcription:
            st.session_state.run_transcription = False
            st.info("Transcription stopped.")
            st.rerun()

    st.markdown("--- ")
    st.info("Press 'Start' to begin. Speak into your microphone.")


# --- CONSTANTS ---
LOG_FILE = "conversation_log.csv"
LANGUAGE_OPTIONS = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta"}

# --- STATE MANAGEMENT ---
def initialize_state():
    if 'mode' not in st.session_state:
        st.session_state.mode = "Real-time Translation"
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

# --- LOGGING ---
def log_to_csv(data):
    df = pd.DataFrame([data])
    file_exists = os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)

# --- AUDIO PROCESSING CLASS ---
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.int16)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_nd = frame.to_ndarray()
        # Assuming mono audio, if stereo, needs adjustment
        audio_segment = (audio_nd * 32767).astype(np.int16)
        self.audio_buffer = np.append(self.audio_buffer, audio_segment)

        # Process when buffer has a decent amount of audio (e.g., 1 second)
        if len(self.audio_buffer) >= frame.sample_rate:
            audio_np = self.audio_buffer.astype(np.float32) / 32768.0
            self.audio_buffer = np.array([], dtype=np.int16) # Clear buffer
            
            text, confidence = transcriber.transcribe_chunk(audio_np)

            if text and confidence > 0.4:
                if st.session_state.mode == "Real-time Translation":
                    self.handle_realtime_translation(text)
                else:
                    self.handle_chat_translation(text)
        return frame

    def handle_realtime_translation(self, text):
        st.session_state.original_text = text
        lang = detect_language(text)
        st.session_state.detected_lang = lang.upper()
        
        # Simple logic: if not English, translate to English. If English, translate to Speaker A's language.
        target_lang = "en" if lang != "en" else st.session_state.speaker_a_lang
        translation = translate_text(text, lang, target_lang)
        st.session_state.translated_text = translation
        
        if translation:
            self.generate_tts(translation, target_lang)
            log_to_csv({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "speaker": "N/A (Real-time)", "original_text": text,
                "source_language": lang, "translated_text": translation,
                "target_language": target_lang
            })

    def handle_chat_translation(self, text):
        speaker = st.session_state.current_speaker
        source_lang = st.session_state.speaker_a_lang if speaker == 'A' else st.session_state.speaker_b_lang
        target_lang = st.session_state.speaker_b_lang if speaker == 'A' else st.session_state.speaker_a_lang

        translation = translate_text(text, source_lang, target_lang)
        
        st.session_state.chat_history.append({
            "speaker": speaker, "original": text, "translated": translation,
            "source_lang": source_lang, "target_lang": target_lang
        })
        
        if translation:
            self.generate_tts(translation, target_lang)
            log_to_csv({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "speaker": speaker, "original_text": text,
                "source_language": source_lang, "translated_text": translation,
                "target_language": target_lang
            })

    def generate_tts(self, text, lang):
        try:
            tts_lang_code = get_tts_language_code(lang)
            audio_bytes = tts_engine.say(text, language=tts_lang_code)
            st.session_state.audio_output = audio_bytes
        except Exception as e:
            st.error(f"TTS Error: {e}")
            st.session_state.audio_output = None

# --- SIDEBAR UI ---
st.sidebar.title("Settings")
st.session_state.mode = st.sidebar.radio("Choose Mode", ["Real-time Translation", "Two-Person Chat"])

speaker_a_lang_name = st.sidebar.selectbox("Speaker A / Default Target Language", list(LANGUAGE_OPTIONS.keys()), index=1)
st.session_state.speaker_a_lang = LANGUAGE_OPTIONS[speaker_a_lang_name]

if st.session_state.mode == "Two-Person Chat":
    speaker_b_lang_name = st.sidebar.selectbox("Speaker B Language", list(LANGUAGE_OPTIONS.keys()), index=0)
    st.session_state.speaker_b_lang = LANGUAGE_OPTIONS[speaker_b_lang_name]

    st.sidebar.subheader("Turn Management")
    if st.sidebar.button(f"Switch to Speaker {'B' if st.session_state.current_speaker == 'A' else 'A'}"):
        st.session_state.current_speaker = 'B' if st.session_state.current_speaker == 'A' else 'A'
    
    current_speaker_lang = speaker_a_lang_name if st.session_state.current_speaker == 'A' else speaker_b_lang_name
    st.sidebar.info(f"**Current Speaker: {st.session_state.current_speaker}** ({current_speaker_lang}) ")

# --- MAIN UI ---
tab1, tab2 = st.tabs(["Translator", "Conversation Log"])

with tab1:
    st.info("Click 'START' below to activate the microphone. Grant permissions if prompted.")
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
                st.audio(st.session_state.audio_output, format="audio/wav", autoplay=True)
                st.session_state.audio_output = None # Clear after playing
    else: # Two-Person Chat Mode
        st.header("Conversation Log")
        for entry in reversed(st.session_state.chat_history):
            align = "flex-start" if entry["speaker"] == 'A' else "flex-end"
            st.markdown(f"<div style='display: flex; justify-content: {align};'>" 
                        f"<div style='background-color: #262730; padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%;'>"
                        f"<b>Speaker {entry['speaker']} ({entry['source_lang']}):</b> {entry['original']}<br>"
                        f"<b>Translation ({entry['target_lang']}):</b> {entry['translated']}"
                        f"</div></div>", unsafe_allow_html=True)

        if st.session_state.audio_output:
            st.audio(st.session_state.audio_output, format="audio/wav", autoplay=True)
            st.session_state.audio_output = None # Clear after playing

with tab2:
    st.header("Full Conversation History")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df, use_container_width=True)
        if st.button("Clear Log"):
            os.remove(LOG_FILE)
            st.rerun()
    else:
        st.info("No conversation history found.")
