import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import librosa
import librosa.display


def get_audio():
    '''
    Ruturns the path to an audio file
    '''

    desired_folder = "Crema"
    root_dir = os.path.dirname(os.path.dirname(__file__))
    folder_path = os.path.join(root_dir, "raw_data", desired_folder)


    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter for WAV files (assuming they have the '.wav' extension)
    wav_files = [file for file in files if file.endswith('.wav')]

    if wav_files:
        # Access the first WAV file
        return os.path.join(folder_path, wav_files[0])
    else:
        return 'There is no audio file in this folder'



def plot_fft(signal, rate, freq_max):
    '''
    Performs fft on .wav file
    '''
    Y = np.abs(scipy.fft.fft(signal))
    X = np.abs(scipy.fft.fftfreq(Y.size) * rate)
    plt.xlim(0, freq_max)
    plt.plot(X, Y)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('Amplitude')
    plt.title('fft')


def plot_spectrogram(signal, rate):
    '''
    computes and plots spectrogram
    '''

    # Compute the Short-Time Fourier Transform (STFT)
    D = librosa.stft(signal)

    # Calculate the magnitude (amplitude) of the STFT
    S = np.abs(D)

    # Convert the magnitude to dB scale
    db_magnitude = librosa.amplitude_to_db(S, ref=np.max)

    # Visualize the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(db_magnitude, sr=rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()



def plot_mfcc(signal, rate, n_mfcc):
    '''
    computes and plots MFCCs
    '''

    # Compute MFCCs with the specified number of coefficients
    mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=n_mfcc)

    # Visualize the MFCCs
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{n_mfcc} MFCCs')
    plt.show()


def plot_mel_spectrogram(signal, rate):
    '''
    computes and plots mel-spectrogram
    '''
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=rate)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.show()


# Load the audio data and sample rate
audio_data, sample_rate = librosa.load(get_audio())
time = np.arange(audio_data.shape[0])/sample_rate

# Create a subplot grid with 2 rows and 1 column
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, audio_data)
plt.xlabel('time(s)')
plt.ylabel('Amplitude')
plt.title('audio file')

plt.subplot(2, 1, 2)
plot_fft(audio_data, sample_rate, 7500)

plt.tight_layout()
plt.show()

plot_spectrogram(audio_data, sample_rate)
plot_mfcc(audio_data, sample_rate, n_mfcc=13)
plot_mel_spectrogram(audio_data, sample_rate)
