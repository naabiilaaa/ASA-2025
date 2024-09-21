import librosa
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt  
import soundfile as sf

def create_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    spect = librosa.feature.melspectrogram(y=y, sr=sr)
    spect_db = librosa.power_to_db(spect, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()
    return spect_db

spectrogram = create_spectrogram('speaker1.wav')
