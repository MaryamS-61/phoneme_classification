import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import requests



st.write("Welcome")
url = 'https://phoneme-service-wifbxua65a-ew.a.run.app'
request = requests.get(url).json()["greeting"]
st.write(request)


# Create a folder to store uploaded audio files (if it doesn't exist)
if not os.path.exists("audio_uploads"):
    os.makedirs("audio_uploads")

# Title
st.title("Audio Spectrogram Viewer")

# Upload an audio file
audio_file = st.file_uploader("Upload an audio file (MP3, WAV, OGG)", type=["mp3", "wav", "ogg"])

# Check if an audio file was uploaded
if audio_file is not None:
    # Save the uploaded audio file to the "audio_uploads" folder
    with open(os.path.join("audio_uploads", audio_file.name), "wb") as f:
        f.write(audio_file.read())
    st.success(f"Uploaded {audio_file.name} to 'audio_uploads' folder.")

    # Generate the spectrogram
    audio_path = os.path.join("audio_uploads", audio_file.name)
    y, sr = librosa.load(audio_path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Display the spectrogram using Matplotlib
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()

    # Display the spectrogram image in Streamlit
    st.pyplot(plt)
else:
    st.warning("Please upload an audio file to view the spectrogram.")
