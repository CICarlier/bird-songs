import os
import glob
from collections import OrderedDict
import librosa
import librosa.display
import soundfile as sf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def loop_and_cut(_id, input_path, output_path, duration=8, max_shape=176400):
    '''
    Cut the audio files to the first 8 seconds and loop through smaller files up to 8 seconds. Writes the transformed
    files to a new folder.
    :param _id: the id of the audio file passed to the function. Should be a 6-digit number
    :param input_path: the path of the folder the audio files is currently stored, e.g. 'audio_noise_reduction/'
    :param output_path: the path of the folder the transformed audio files should be saved at, e.g. 'audio_8sec/'
    :param duration: the length of the output file. Default is 8 seconds
    :param max_shape: the max length of the x array of loaded files. Default is 176400 (8 seconds)
    :return: no returned values, just a print statement
    '''
    x, sr = librosa.load(f'{input_path}{_id}.mp3', duration=duration)
    if x.shape[0] < max_shape:
        while x.shape[0] < max_shape:
            y, sr = librosa.load(f'{input_path}{_id}.mp3')
            x = np.append(x, y)
        sf.write(f'audio_noise_reduction/{_id}.wav', x, sr)
        z, r = librosa.load(f'audio_noise_reduction/{_id}.wav', duration=duration)
        sf.write(f'{output_path}/{_id}.wav', z, sr)
        print(f'looped and cut {_id}')
    else:
        sf.write(f'{output_path}/{_id}.wav', x, sr)
        print(f'cut {_id}')


def find_best_clip(y, sample_rate, subclip_sec):
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

    if subclip_sec < 0 or subclip_sec > 50:
        return "Error: You must specify a subclip between 0 and 50 seconds"
        pass

    # linear rms level and scaling factor
    rms_level_db = 0  # the 0dB RMS level is equivalent to an amplitude of 1
    sig = y
    r = 10 ** (rms_level_db / 20.0)
    a = np.sqrt((len(sig) * r ** 2) / np.sum(sig ** 2))
    # Normalized amplitude signal
    y_norm = y * a

    # Calc length of audio clip and sub-clip in samples (i.e. not seconds)
    audio_length = len(y_norm)
    subclip_length = int(subclip_sec * sample_rate)

    # Check if y is shorter than the subclip_sec and if so, wrap y until length = subclip_sec
    if audio_length < subclip_length:
        number_repeats = subclip_length // audio_length
        remaining_samples = subclip_length % audio_length
        # Create new audio clip y by repeat
        y_new_repeat = np.tile(y_norm, number_repeats)
        if remaining_samples > 0:
            y_new_remaining = y_norm[:remaining_samples]
            y_new = np.append(y_new_repeat, y_new_remaining, axis=0)
        else:
            y_new = y_new_repeat

        # Update start and stop index of repeated audio file
        return 0, len(y_new) + 1, y_new


    else:
        # Calculate area of window for each hop along the audio waveform
        hop_stride = int(min(subclip_length * sample_rate / 5, audio_length / 20))
        total_hops = int(audio_length / hop_stride)

        # Store data on each hop
        hop_data = OrderedDict()
        hop_window_start = 0
        hop_window_end = subclip_length + 1
        hop = 0

        # Clip negative amplitude values for area calculation for each hop
        y_norm_positive = y_norm.clip(min=0)

        # Keep hopping until just before a hop would overlap past the end of the audio file
        while hop_window_end <= audio_length:
            y_window = y_norm_positive[hop_window_start:hop_window_end]
            hop_window_area = np.trapz(y_window, dx=1 / sample_rate, axis=0)
            hop_data[hop] = [hop_window_start, hop_window_end, hop_window_area]
            hop_window_start += hop_stride
            hop_window_end += hop_stride
            hop += 1

        # Add one window hop to cover remaining area at end of file if skipped above
        if hop_window_end > audio_length:
            hop_window_start = audio_length - subclip_length
            y_window = y_norm_positive[hop_window_start:]
            hop_window_area = np.trapz(y_window, dx=1 / sample_rate, axis=0)
            hop_data[hop] = [hop_window_start, hop_window_end, hop_window_area]

        # Find hop with maximum area under the waveform.
        maxhop = max(hop_data, key=lambda x: hop_data[x][-1])
        max_y_window = y_norm_positive[hop_data[maxhop][0]:hop_data[maxhop][1]]
        max_start_window_seconds = hop_data[maxhop][0] / sample_rate

        # Store start and stop of sub-clip with max area
        start = hop_data[maxhop][0]
        end = hop_data[maxhop][1]

        return start, end, y



def mel_spectograms(audio_file, path, _id, best_clip=False, subclip_sec=0):
    '''
    Generate mel-spectrograms images without borders
    :param audio_file: path of the audio file to load e.g. audio_8sec/169075.wav
    :param path: name of folder to save output file e.g. images/audio_8sec
    :param _id: id of the file processed - should be a 6-figure integer
    :param best_clip: whether to pass the dataset to the find_best_clip function or not. Default is False.
    :param num_div: number of divisions to pass to the find_best_clip function.
    :return: no value returned, image saved to a folder.
    '''

    y, sr = librosa.load(audio_file)
    if best_clip == False:
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
    # Clip file to best subclip, if requested
    else:
        print(_id)
        start, end, new_y = find_best_clip(y, sr, subclip_sec=subclip_sec) # new_y returns looped array if file length is less than subclip_sec
        y_norm = new_y[start:end]
        mel = librosa.feature.melspectrogram(y_norm, sr)
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
    if best_clip:
        id_only = _id.split('-')[-1]
        plt.savefig(f'images/{path}/{id_only}.jpg', dpi=height)
        plt.close()
    else:
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

# Generate 8-second audio files based on the resampled files
clear_directory('audio_8sec')
for file in glob.glob("audio_noise_reduction/*"):
    _id = file.split('.')[0].split('-')[-1]
    loop_and_cut(_id, 'audio_noise_reduction/resampled-clean-', 'audio_8sec')

# Generate mel-spectrograms of the 8-second files
clear_directory('images/mel_spectrograms_8sec')
for file in glob.glob("audio_8sec/*"):
    _id = file.split('\\')[1].split('.')[0]
    mel_spectograms(file, 'mel_spectrograms_8sec', _id)


# Generate 8-second audio files based on the original files
clear_directory('audio_8sec_unprocessed')
for file in glob.glob("audio/*"):
    _id = file.split('\\')[1].split('.')[0]
    loop_and_cut(_id, 'audio/', 'audio_8sec_unprocessed')

# Generate mel-spectrograms of the 8-second files
clear_directory('images/mel_spectrograms_8sec_unprocessed')
for file in glob.glob("audio_8sec_unprocessed/*"):
    _id = file.split('\\')[1].split('.')[0]
    mel_spectograms(file, 'mel_spectrograms_8sec_unprocessed', _id)

# Generate mel-spectrograms of the best clip of each files
clear_directory('images/mel_spectrograms_best_clip')
for file in glob.glob("audio_noise_reduction/*"):
    _id = file.split('\\')[1].split('.')[0]
    mel_spectograms(file, 'mel_spectrograms_best_clip', _id, best_clip=True, subclip_sec=1.5)


# Generate 2-second audio files based on the resampled files with no silence
clear_directory('audio_2sec_no_silence')
for file in glob.glob("audio_no_silence/*"):
    _id = file.split('.')[0].split('-')[-1]
    loop_and_cut(_id, 'audio_no_silence/no-silence-resampled-clean-', 'audio_2sec_no_silence', duration=2, max_shape=44100)

# Generate mel-spectrograms of the 2-second files
clear_directory('images/mel_spectrograms_2sec_no_silence')
for file in glob.glob("audio_2sec_no_silence/*"):
    _id = file.split('\\')[1].split('.')[0]
    mel_spectograms(file, 'mel_spectrograms_2sec_no_silence', _id)

# Generate mel_spectograms of augmented data
clear_directory('images/mel_spectrograms_augmented_data')
for file in glob.glob("audio_augmentation/*"):
    _id = file.split('\\')[1].split('.')[0]
    mel_spectograms(file, 'mel_spectrograms_augmented_data', _id)
