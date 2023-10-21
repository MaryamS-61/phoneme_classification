from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
import wave
from phoneme.ml.preprocessing import compute_spectrogram, normalization, zero_padding
import numpy as np
import os
import pickle
from keras.models import load_model


app = FastAPI()

<<<<<<< HEAD
@app.get("/predict")
def predict():
    return {"greeting":"Hello World!"}
=======
@app.post("/")
async def predict(bytes: bytes=File(...)):
    #      with open(os.path.join("audio_uploads", bytes.name), "wb") as f:
    #     audio_bytes = bytes.read()
    #     print(bytes.name)
    #     print(audio_bytes)

    # bytes = bytes.decode('utf-8')
    # print(bytes)
    # output_wav_file = 'output.wav'

    # # Define the WAV file parameters
    # nchannels = 1  # Mono audio
    # sampwidth = 2  # 2 bytes per sample (16-bit)
    # framerate = 44100  # Sample rate in Hz
    # nframes = len(bytes) // (sampwidth * nchannels)  # Number of frames
    # comptype = 'NONE'
    # compname = 'not compressed'

    # # Open a WAV file for writing
    # with wave.open(output_wav_file, 'wb') as wav_file:
    #     wav_file.setnchannels(nchannels)
    #     wav_file.setsampwidth(sampwidth)
    #     wav_file.setframerate(framerate)
    #     wav_file.setnframes(nframes)
    #     wav_file.setcomptype(comptype, compname)
        
    #     # Write the audio data to the WAV file
    #     wav_file.writeframes(bytes)
    # #compute_spectrogram(wav_file)
    # wav_file = np.frombuffer(bytes, dtype=np.float32)
    # if not np.isfinite(wav_file).all():
    # # Handle or clean the non-finite values
    #     wav_file = np.nan_to_num(wav_file)
    # spectrogram = compute_spectrogram(wav_file)
    # normalized_spectrogram = normalization(spectrogram)
    # #print(spectrogram)
    # cnn_input = zero_padding(normalized_spectrogram)
    # print(cnn_input)
    # print(cnn_input.shape)
    # print(spectrogram)
    
    file = "/home/m7rudloff/code/MaryamS-61/phoneme_classification/Crema_happy_cnn_input.pkl"
    with open(file, 'rb') as file:
        loaded_df = pickle.load(file)
    print(loaded_df.SpecInputCNN)
    app.state.model = load_model("phoneme/api/first_cnn_model_maryam.h5")
    x = np.expand_dims(loaded_df.SpecInputCNN, axis=0)
    y_pred = app.state.model.predict(x)
    classes = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    result = classes[np.argmax(y_pred[0])]
    return {"result":result}
>>>>>>> 48c962f (trying to merge everything)
