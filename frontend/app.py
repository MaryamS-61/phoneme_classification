import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
#from phoneme.ml.preprocessing import compute_spectrogram, normalization, zero_padding
from keras.models import load_model
import io


# Do not do this
def compute_spectrogram(wav_file):

    #y, sr = librosa.load(wav_file, sr=8000)
    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(wav_file)

    # Calculate the magnitude (amplitude) of the STFT
    S = np.abs(D)

    # Convert the magnitude to dB scale
    db_magnitude = librosa.amplitude_to_db(S, ref=np.max)
    return db_magnitude



def normalization(spectrogram):
    '''
    this function normalizes the values of spectrogram
    '''
    original_min = np.min(spectrogram)
    original_max = np.max(spectrogram)

    # Define the new min and max values
    new_min = 0
    new_max = 1

    # Normalize the data to the new range
    normalized_x = ((spectrogram - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min
    return normalized_x


def zero_padding(data):
    '''
    This functiuon first zero-pads the input spectrogram to maximum width of the dataset
    and then zero-pads a mask of 3x3  to the whole spectrogram image.
    '''
    max_width = 79 # this is the maximum spectrogram width of the whole Crema dataset.
    zero_padded_matrix = np.pad(data, ((0, 0), (0, max_width - data.shape[1])))

    # Define the number of rows and columns to add around the images
    pad_rows = 3
    pad_cols = 3

    # Calculate the new shape of the zero-padded array
    new_shape = (
        zero_padded_matrix.shape[0] + 2 * pad_rows,
        zero_padded_matrix.shape[1] + 2 * pad_cols
    )

    # Create a zero-padded array with the new shape
    zero_padded_mask = np.zeros(new_shape)

    # Copy the original array into the center of the zero-padded array
    zero_padded_mask[pad_rows:-pad_rows, pad_cols:-pad_cols] = zero_padded_matrix
    return zero_padded_mask


# Define the emotions dictionary at the global level
emotions = {
    "ANG": ("Anger 😡", "angry.gif"),  # Add the path to your angry emoji GIF
    "DIS": ("Disgust 🤢", "disgust.gif"),  # Add the path to your disgust emoji GIF
    "FEA": ("Fear 😱", "fear.gif"),  # Add the path to your fear emoji GIF
    "HAP": ("Happiness 😀", "happy.gif"),  # Add the path to your happy emoji GIF
    "NEU": ("Neutral 😐", "neutral.gif"),  # Add the path to your neutral emoji GIF
    "SAD": ("Sadness 😭", "sad.gif")  # Add the path to your sad emoji GIF
}

# Define the HTML and CSS styling for the fixed header
header_html = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300&display=swap');
        .header-container {
            background-color: #B22222;
            padding: 35px; /* Increase padding to make it taller */
            text-align: center;
            width: 37%;
            color: #FFFFFF; /* Set text color to white */
            height: 150px; /* Adjust height to make it taller */
            position: fixed; /* Make the header fixed at the top */
            top: 0; /* Stick it to the top */
            z-index: 1000; /* Adjust the z-index if needed */
        }
        .header-title {
            font-family: 'Merriweather', serif;
            font-size: 28px;
            color: #FFFFFF; /* Set text color to white */
        }
        .header-subtitle {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
            color: #FFFFFF; /* Set text color to white */
        }
        .content {
            margin-top: 170px; /* Adjust the margin to move content below the header */
        }
    </style>
    <div class="header-container">
        <h1 class="header-title">Emotion Speech Recognition</h1>
        <h3 class="header-subtitle">Data Science Le Wagon Bootcamp Part-time</h3>
    </div>
"""

# Define the HTML and CSS styling for the footer
footer_html = """
    <style>
        .footer-container {
            background-color: #B22222;
            padding: 10px;
            text-align: center;
            width: 37%;
            color: #FFFFFF;
            position: fixed;
            bottom: 0;
            display: flex;
            justify-content: center;
        }
    </style>
    <div class="footer-container">batch-1283 Le Wagon</div>
"""

# Define the CSS styling for the fixed sidebar with the image
sidebar_html = """
    <style>
        .fixed-sidebar {
            position: fixed;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
        }
    </style>
"""

# Define the prediction_emotion_feedback function
def prediction_emotion_feedback(prediction):
    if prediction in emotions:
        emotion_name, emoji_path = emotions[prediction]
        st.write("The detected emotion in the audio is:", emotion_name)

        # Display the spectrogram using Matplotlib
        st.pyplot(plt)

        # Display the emoji and emotion name in the fixed sidebar
        st.sidebar.image(emoji_path, caption=emotion_name, use_column_width=True, output_format='auto')

# Render the HTML in Streamlit
st.markdown(header_html, unsafe_allow_html=True)

# Add content below the header
st.write("This is your content below the header.")

# Width of the original image (replace with the actual width)
original_width = 800

# Calculate half of the original width
half_width = original_width // 5

# Create a centered sidebar
st.sidebar.image("lewagon.png", width=half_width)


# List of dummy names
names = ["Maryam Sadreddini", "Sina Naghizadeh", "Anna Snizhko", "Martin Jahr"]

# Define the HTML and CSS styling for centering, making names bold, and increasing font size
style_html = """
    <style>
        .sidebar-names {
            text-align: center;
            font-weight: bold;
            font-size: 16px;
        }
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>

"""

# Render the HTML in Streamlit to apply CSS styling
st.sidebar.markdown(style_html, unsafe_allow_html=True)

# Display the names in the sidebar with the applied styling
for name in names:
    st.sidebar.markdown(f"<p class='sidebar-names'>{name}</p>", unsafe_allow_html=True)


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
        audio_data = audio_file.read()
        # url = 'https://phoneme-service-wifbxua65a-ew.a.run.app'
        #url = "http://localhost:8000/predict_for_real"
        #request = requests.post(url,  files = {"sound":audio_data})
        #prediction = json.loads(request.text)['result']

        wav_file , _ = librosa.load(io.BytesIO(audio_data), sr=8000)
        spectrogram = compute_spectrogram(wav_file)
        normalized_spectrogram = normalization(spectrogram)
        cnn_input = zero_padding(normalized_spectrogram)

        # load model
        model = load_model("first_cnn_model_maryam.h5")
        x = np.expand_dims(cnn_input, axis=0)

        # predict
        y_pred = model.predict(x)
        classes = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
        prediction = classes[np.argmax(y_pred[0])]

        f.write(audio_data)
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


    if prediction in emotions:
        prediction_emotion_feedback(prediction)

# Render the HTML in Streamlit for the footer
st.markdown(footer_html, unsafe_allow_html=True)

# Render the CSS for the fixed sidebar
st.markdown(sidebar_html, unsafe_allow_html=True)
