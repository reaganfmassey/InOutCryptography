import librosa
import numpy as np
import hashlib
from cryptography.fernet import Fernet

def extract_key_changes(audio_file):
    y, sr = librosa.load(audio_file)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_changes = []

    # Detect key changes by measuring chroma similarity over time
    prev_key = np.argmax(np.mean(chroma, axis=1))
    for i in range(1, chroma.shape[1], 100):  # Sliding window
        curr_key = np.argmax(np.mean(chroma[:, i:i+100], axis=1))
        if curr_key != prev_key:
            key_changes.append(curr_key)
            prev_key = curr_key

    return key_changes

# Example usage
key_changes = extract_key_changes("Giant_Steps.wav")
print("Detected Key Changes:", key_changes)
