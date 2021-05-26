import os
import glob
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def loop_and_cut(_id, input_path, output_path):
    '''
    Cut the audio files to the first 8 seconds and loop through smaller files up to 8 seconds. Writes the transformed
    files to a new folder.
    :param _id: the id of the audio file passed to the function. Should be a 6-digit number
    :param input_path: the path of the folder the audio files is currently stored, e.g. 'audio_noise_reduction/'
    :param output_path: the path of the folder the transformed audio files should be saved at, e.g. 'audio_8sec/'
    :return: no returned values, just a print statement
    '''
    x, sr = librosa.load(f'{input_path}{_id}.mp3', duration=8)
    if x.shape[0] < 176400: # 8-sec file will have a shape of (176400,)
        while x.shape[0] < 176400:
            y, sr = librosa.load(f'{input_path}{_id}.mp3')
            x = np.append(x, y)
        sf.write(f'audio_noise_reduction/{_id}.wav', x, sr)
        z, r = librosa.load(f'audio_noise_reduction/{_id}.wav', duration=8)
        sf.write(f'{output_path}/{_id}.wav', z, sr)
        print(f'looped and cut {_id}')
    else:
        sf.write(f'{output_path}/{_id}.wav', x, sr)
        print(f'cut {_id}')


def find_best_clip(y, num_div=1, sample_rate = 22050):
    '''
    Performs Root-Mean-Square (RMS) Amplitude Normalization on the audio files, divides the file into multiple window
     then selects the best clip based on maximum amplitude.
    Why use RMS?
        Since an audio signal can have both positive and negative amplitude values, if we took the arithmetic mean of a
        sine wave, the negative values would offset the positive values and the result would be zero. This is where the
        RMS level can be useful as it is calculated by squaring each sample value (so they are all positive), then
        calculating the square root signal average.
        To normalize the amplitude of a signal is based on the RMS amplitude, we multiply a scaling factor, a, by the
        sample values in our signal to change the amplitude such that the result has the desired RMS level, R.
        See https://www.hackaudio.com/digital-signal-processing/amplitude/rms-normalization/ for reference.
    :param y: the array representing an audio file loaded with librosa
    :param num_div: number of file divisions to perform. Must be 1, 2, 4 or 8.
    :return: dataframe with start and end times of best clips for each file.
    '''

    if num_div not in [1, 2, 4, 8]:
        print("Error: You must only select 1, 2, 4, or 8 divisions")
        pass

    # linear rms level and scaling factor
    rms_level_db = 0 # the 0dB RMS level is equivalent to an amplitude of 1
    sig = y
    r = 10 ** (rms_level_db / 20.0)
    a = np.sqrt((len(sig) * r ** 2) / np.sum(sig ** 2))
    # Normalized amplitude signal
    y_norm = y * a

    # Clip negative amplitude values for area calculation
    y_norm_positive = y_norm.clip(min=0)

    # Calculate areas for each window
    length_audio_clip = int(len(y_norm_positive) / sample_rate) # seconds
    print(length_audio_clip)
    duration_per_window = length_audio_clip / num_div  # seconds
    hop_length = duration_per_window / 10  # seconds
    total_hops = int(length_audio_clip / hop_length)

    area = []
    for hop in range(total_hops):
        start_window = int(hop * hop_length * sample_rate)
        end_window = int((hop + 1) * hop_length * sample_rate) + 1
        y_window = y_norm_positive[start_window:end_window]
        area_window = np.trapz(y_window, dx=1 / sample_rate, axis=0)
        area.append(area_window)

    max_window = np.argmax(area)
    max_start_window = int(max_window * hop_length * sample_rate)
    max_end_window = int((max_window + 1) * hop_length * sample_rate) + 1

    return max_start_window, max_end_window



def mel_spectograms(audio_file, path, _id, best_clip=False, num_div=0):
    '''
    Generate mel-spectrograms images without borders
    :param audio_file: path of the audio file to load e.g. audio_8sec/169075.wav
    :param path: name of folder to save output file e.g. images/audio_8sec
    :param _id: id of the file processed - should be a 6-figure integer
    :return: no value returned, image saved to a folder.
    '''

    y, sr = librosa.load(audio_file)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    # Clip file to best subclip, if requested
    if best_clip == True:
        print(_id)
        start, end = find_best_clip(y, num_div=num_div)
        y_norm = y[start:end]
        mel = librosa.feature.melspectrogram(y_norm, sr, n_mels=n_mels,
                                              n_fft=n_fft, hop_length=hop_length, fmin=fmin)
    m_db = librosa.power_to_db(mel, ref=np.max)

    sizes = np.shape(m_db)
    height = float(sizes[0])
    width = float(sizes[1])
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    img = librosa.display.specshow(m_db, y_axis='mel', x_axis='time', sr=sr, ax=ax, cmap='bwr')
    plt.savefig(f'images/{path}/{_id}.jpg', dpi=height)
    plt.close()


def clear_directory(directory):
    if os.path.exists(directory):
        files = glob.glob(f'{directory}/*')
        for f in files:
            print(f"Deleting {f}")
            os.remove(f)
    else:
        os.makedirs(directory)


# Import the filtered features file
features = pd.read_csv('features_filtered.csv')

n_mels = 20
fmin = 4000 # Minimum Hz
n_fft = 512 # Choose 2^n where n is integer
hop_length = 256 # Choose equal, half, or quarter of N_FFT

# # Generate 8-second audio files based on the resampled files
# clear_directory('audio_8sec')
# for file in glob.glob("audio_noise_reduction/*"):
#     _id = file.split('.')[0].split('-')[-1]
#     loop_and_cut(_id, 'audio_noise_reduction/resampled-clean-', 'audio_8sec')
#
# # Generate mel-spectrograms of the 8-second files
# clear_directory('images/mel_spectrograms_8sec')
# for file in glob.glob("audio_8sec/*"):
#     _id = file.split('\\')[1].split('.')[0]
#     mel_spectograms(file, 'mel_spectrograms_8sec', _id)
#
#
# # Generate 8-second audio files based on the original files
# clear_directory('audio_8sec_unprocessed')
# for file in glob.glob("audio/*"):
#     _id = file.split('\\')[1].split('.')[0]
#     loop_and_cut(_id, 'audio/', 'audio_8sec_unprocessed')
#
# # Generate mel-spectrograms of the 8-second files
# clear_directory('images/mel_spectrograms_8sec_unprocessed')
# for file in glob.glob("audio_8sec_unprocessed/*"):
#     _id = file.split('\\')[1].split('.')[0]
#     mel_spectograms(file, 'mel_spectrograms_8sec_unprocessed', _id)

# Generate mel-spectrograms of the best clip of each files
clear_directory('images/mel_spectrograms_best_clip')
for file in glob.glob("audio_noise_reduction/*"):
    _id = file.split('\\')[1].split('.')[0]
    mel_spectograms(file, 'mel_spectrograms_best_clip', _id, best_clip=True, num_div=4)