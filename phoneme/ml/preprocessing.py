import numpy as np
import pandas as pd
import librosa
#from one_sample_preprocessing import get_audio


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

#wav_file = get_audio()
