from fastapi import FastAPI, File
from phoneme.ml.preprocessing import compute_spectrogram, normalization, zero_padding
import numpy as np
import pickle
from keras.models import load_model
import librosa
import io


app = FastAPI()

@app.post("/predict_fake")
async def predict(bytes: bytes=File(...)):
    file = "Crema_happy_cnn_input.pkl"
    with open(file, 'rb') as file:
        loaded_df = pickle.load(file)
    print(loaded_df.SpecInputCNN)
    app.state.model = load_model("phoneme/api/first_cnn_model_maryam.h5")
    x = np.expand_dims(loaded_df.SpecInputCNN, axis=0)
    print(x.shape)
    y_pred = app.state.model.predict(x)
    classes = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    result = classes[np.argmax(y_pred[0])]
    return {"result":result}


@app.post("/predict_for_real")
async def upload_file(sound: bytes = File(...)):
    wav_file , _ = librosa.load(io.BytesIO(sound), sr=8000)
    spectrogram = compute_spectrogram(wav_file)
    normalized_spectrogram = normalization(spectrogram)
    cnn_input = zero_padding(normalized_spectrogram)
    app.state.model = load_model("phoneme/api/first_cnn_model_maryam.h5")
    x = np.expand_dims(cnn_input, axis=0)
    print(x.shape)
    y_pred = app.state.model.predict(x)
    classes = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    result = classes[np.argmax(y_pred[0])]
    return {"result":result}
