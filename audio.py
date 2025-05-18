import os
import tempfile
import speech_recognition as sr
from gtts import gTTS
import pygame
from typing import Optional, Tuple
import time

class AudioManager:
    def __init__(self):
        """Initialize the AudioManager with speech recognition and audio playback capabilities."""
        self.recognizer = sr.Recognizer()
        pygame.mixer.init()
        self._current_audio_file = None
        self._is_playing = False

    def record_audio(self, timeout: int = 5) -> Tuple[bool, str]:
        """
        Record audio from the microphone and convert it to text.
        
        Args:
            timeout (int): Maximum time to listen for speech in seconds
            
        Returns:
            Tuple[bool, str]: (success, result)
                - If successful: (True, recognized_text)
                - If failed: (False, error_message)
        """
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for audio input
                audio = self.recognizer.listen(source, timeout=timeout)
                
            # Convert speech to text using Google's API
            text = self.recognizer.recognize_google(audio)
            return True, text
            
        except sr.WaitTimeoutError:
            return False, "No speech detected within timeout period"
        except sr.RequestError:
            return False, "Could not connect to the speech recognition service"
        except sr.UnknownValueError:
            return False, "Could not understand the audio"
        except Exception as e:
            return False, f"An error occurred: {str(e)}"

    def speak_text(self, text: str, lang: str = 'en') -> Tuple[bool, str]:
        """
        Convert text to speech and play it.
        
        Args:
            text (str): The text to convert to speech
            lang (str): Language code (default: 'en')
            
        Returns:
            Tuple[bool, str]: (success, message)
                - If successful: (True, "Audio played successfully")
                - If failed: (False, error_message)
        """
        try:
            if not text or not text.strip():
                return False, "No text provided"

            # Clean up previous audio file if it exists
            self._cleanup_audio()
            
            # Create temporary file for audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            self._current_audio_file = temp_file.name
            temp_file.close()
            
            try:
                # Generate speech
                tts = gTTS(text=text.strip(), lang=lang)
                tts.save(self._current_audio_file)
            except Exception as e:
                return False, f"Failed to generate speech: {str(e)}"
            
            try:
                # Initialize mixer if needed
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                
                # Play audio
                pygame.mixer.music.load(self._current_audio_file)
                pygame.mixer.music.play()
                self._is_playing = True
                
                # Give a short time for playback to start
                time.sleep(0.1)
                
                if not pygame.mixer.music.get_busy():
                    return False, "Failed to start audio playback"
                    
                return True, "Audio playing"
                
            except Exception as e:
                return False, f"Failed to play audio: {str(e)}"
            
        except Exception as e:
            self._cleanup_audio()
            return False, f"Failed to process audio: {str(e)}"

    def stop_audio(self) -> Tuple[bool, str]:
        """
        Stop any currently playing audio.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                self._is_playing = False
                self._cleanup_audio()
                return True, "Audio stopped"
            return False, "Audio system not initialized"
        except Exception as e:
            return False, f"Failed to stop audio: {str(e)}"

    def _cleanup_audio(self) -> None:
        """Clean up temporary audio files."""
        if self._current_audio_file and os.path.exists(self._current_audio_file):
            try:
                if self._is_playing:
                    pygame.mixer.music.stop()
                    self._is_playing = False
                os.unlink(self._current_audio_file)
            except:
                pass  # Ignore cleanup errors
            self._current_audio_file = None

    def __del__(self):
        """Cleanup on object destruction."""
        self._cleanup_audio()
        if pygame.mixer.get_init():
            pygame.mixer.quit() 