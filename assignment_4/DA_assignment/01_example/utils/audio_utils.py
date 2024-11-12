########################################################################
# import additional python-library
########################################################################
import librosa.core
import librosa.feature
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

import os
import sys
import librosa

# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.
    wav_name : str
        target .wav file
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data
    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data

        return sr, np.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        logger.warning(f'{msg}')


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0):
    """
    convert file_name to a vector array.
    file_name : str
        target .wav file
    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    # calculate total vector size
    n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

    # skip too short clips
    if n_vectors < 1:
        return np.empty((0, dims))

    # generate feature vectors by concatenating multiframes
    vectors = np.zeros((n_vectors, dims))
    for t in range(n_frames):
        vectors[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + n_vectors].T

    return vectors


########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_data(file_list,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vectors = file_to_vectors(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros((len(file_list) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data

class MIMIIFan_Train(Dataset):
    def __init__(
        self, 
        root: str,
        machine: str,
        train=True,
        transform=None,
        target_transform=None,
        n_mels=64,
        n_frames=32,
        n_hop_frames=16,
        n_fft=1024,
        hop_length=512,
        power=2.0):

        id_label = 4
        id_domain = 2

        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_hop_frames = n_hop_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power

        self.audio_path = root + machine + '/'+ 'test'
        self.audio_files = np.array(os.listdir(self.audio_path))
        metadata = np.array([f.split("_")[:5] for f in self.audio_files])
        self.labels = (metadata[:,id_label] == 'anomaly')

        mask = metadata[:,id_domain] == ('source' if train else 'target')
        self.audio_files = self.audio_files[mask]
        self.labels = self.labels[mask]
            
        self.transform = transform
        self.target_transform = target_transform

        self.data = self._process_files()
        # self.data = self._process_files()

    def __len__(self):
        return len(self.data)


    def _process_files(self):

        data = []
        for id_f in range(len(self.audio_files)):
            file_path = os.path.join(self.audio_path, self.audio_files[id_f])
            label = self.labels[id_f].astype(np.int64)

            dims = self.n_mels * self.n_frames

            if self.transform:
                f = self.transform(file_path)
            else:
            # iterate file_to_vector_array()
                vectors = file_to_vectors(file_path,
                n_mels=self.n_mels,
                n_frames=self.n_frames,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=self.power)
                vectors = (vectors.reshape(-1, self.n_mels,self.n_frames)-25)/0.4
                #f = vectors
                f = vectors[::self.n_hop_frames, :].astype(np.float32)
                # if id_f == 0:
                #     self.data = np.zeros((len(self.audio_files) * vectors.shape[0], dims), float)
                # self.data[vectors.shape[0] * id_f : vectors.shape[0] * (id_f + 1), :] = vectors
                label =  label*np.ones(f.shape[0])
                
            if self.target_transform:
                label = self.target_transform(label)
            
            data.extend([(f,l) for f,l in zip(f, label)])
        return data

        
    def __getitem__(self, idx):
        return self.data[idx]

    

class MIMIIFan_Test(Dataset):
    def __init__(
        self, 
        root: str,
        machine: str,
        train=True,
        transform=None,
        target_transform=None,
        n_mels=64,
        n_frames=32,
        n_hop_frames=8,
        n_fft=1024,
        hop_length=512,
        power=2.0):

        id_label = 4
        id_domain = 2

        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_hop_frames = n_hop_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power

        self.audio_path = root + machine + '/'+ 'test'
        self.audio_files = np.array(os.listdir(self.audio_path))
        metadata = np.array([f.split("_")[:5] for f in self.audio_files])
        self.labels = (metadata[:,id_label] == 'anomaly')

        mask = metadata[:,id_domain] == ('source' if train else 'target')
        self.audio_files = self.audio_files[mask]
        self.labels = self.labels[mask]
            
        self.transform = transform
        self.target_transform = target_transform

        self.data = self._process_files()
        # self.data = self._process_files()

    def __len__(self):
        return len(self.data)


    def _process_files(self):

        data = []
        for id_f in range(len(self.audio_files)):
            file_path = os.path.join(self.audio_path, self.audio_files[id_f])
            label = self.labels[id_f].astype(np.int64)

            dims = self.n_mels * self.n_frames

            if self.transform:
                f = self.transform(file_path)
            else:
            # iterate file_to_vector_array()
                vectors = file_to_vectors(file_path,
                n_mels=self.n_mels,
                n_frames=self.n_frames,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                power=self.power)
                vectors = (vectors.reshape(-1, self.n_mels,self.n_frames)-25)/0.4
                #f = vectors
                f = vectors[::self.n_hop_frames, :].astype(np.float32)
                # if id_f == 0:
                #     self.data = np.zeros((len(self.audio_files) * vectors.shape[0], dims), float)
                # self.data[vectors.shape[0] * id_f : vectors.shape[0] * (id_f + 1), :] = vectors
                label =  label*np.ones(f.shape[0])
                
            if self.target_transform:
                label = self.target_transform(label)
            
            data.append((f,label))
        return data

        
    def __getitem__(self, idx):
        return self.data[idx]

    

    