#!/usr/bin/env python3
"""
Demo script to test bidirectional translation functionality
"""

from main import translate_text, detect_language, get_tts_language_code

def demo_bidirectional_translation():
    """Demonstrate bidirectional translation capabilities."""
    
    print("🔄 BIDIRECTIONAL TRANSLATION DEMO")
    print("=" * 40)
    
    # Test cases
    test_cases = [
        # Hindi to English
        ("नमस्ते, आप कैसे हैं?", "hi"),
        # Tamil to English  
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "ta"),
        # English to Hindi
        ("Hello, how are you?", "en"),
        # English to Tamil
        ("Good morning, have a nice day!", "en"),
    ]
    
    for text, detected_lang in test_cases:
        print(f"\n📝 Input: [{detected_lang.upper()}] {text}")
        
        if detected_lang in ["hi", "ta", "mr"]:
            # Indian language → English
            translation = translate_text(text, detected_lang, "en")
            tts_lang = "en"
            print(f"🔄 Translation: [{detected_lang.upper()}] → [EN] {translation}")
            
        elif detected_lang == "en":
            # English → Hindi (default) or Tamil
            for target in ["hi", "ta"]:
                translation = translate_text(text, "en", target)
                tts_lang = get_tts_language_code(target)
                target_name = "Hindi" if target == "hi" else "Tamil"
                print(f"🔄 Translation: [EN] → [{target.upper()}] {translation}")
                print(f"🔊 TTS Language: {tts_lang}")
        
        print("-" * 40)

if __name__ == "__main__":
    demo_bidirectional_translation()
