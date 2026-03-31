# app/audio_processor.py

import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, n_mfcc=13):
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_data, sr=22050):
        """Extract MFCC features from audio data"""
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=self.n_mfcc)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            return mfccs_processed
        except Exception as e:
            print(f"Audio feature extraction error: {e}")
            return None

    def record_audio(self, duration=3, sr=22050):
        """Record audio from microphone"""
        import sounddevice as sd
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()  # Wait until recording is finished
        return audio.flatten()