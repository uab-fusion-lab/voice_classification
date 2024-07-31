
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch

import torchaudio.transforms as T

class AmplitudeNormalization:
    def __call__(self, waveform):
        # Normalize the waveform to be within [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        return waveform

# To use it:
# waveform, sample_rate = torchaudio.load('path/to/audio.wav')
# waveform = AmplitudeNormalization()(waveform)


class PadTrimAudio:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, waveform):
        if waveform.size(1) > self.max_len:
            # Trim the waveform if longer than max_len
            waveform = waveform[:, :self.max_len]
        elif waveform.size(1) < self.max_len:
            # Pad with zeros if shorter than max_len
            padding_size = self.max_len - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding_size), "constant", 0)
        return waveform

from sklearn.preprocessing import StandardScaler
import numpy as np

class FeatureNormalization:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, features):
        # Fit the scaler on the training set features
        self.scaler.fit(features)

    def transform(self, features):
        # Apply normalization to features
        return self.scaler.transform(features)

class SoundDataset(Dataset):
    def __init__(self, data_dir, labels_df, transform=None, max_len=1000000):
        self.data_dir = data_dir
        self.labels_df = labels_df
        self.transform = transform
        self.max_len = max_len
        self.orderlist = ['RLP.wav', 'RUP.wav', 'RUA Hum.wav', 'LUA Hum.wav', 'RUA.wav', 'RMP.wav', 'LMP.wav', 'LUA.wav', 'LLP.wav', 'LUP.wav']

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        patient_id = str(idx+1).zfill(3)
        audio_dir = os.path.join(self.data_dir, str(patient_id), 'breath Eko')
        audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

        # Concatenate audio files
        waveform_list = [torch.zeros(1, self.max_len) for _ in range(len(self.orderlist))]
        for audio_file in audio_files:
            file_name = os.path.basename(audio_file)
            index = self.orderlist.index(file_name)
            waveform, sample_rate = torchaudio.load(audio_file)
            if self.transform:
                waveform = self.transform(waveform)
            waveform = PadTrimAudio(self.max_len)(waveform)
            waveform_list[index] = waveform


        # Concatenate all waveforms along the time dimension
        waveform = torch.cat(waveform_list, dim=0)
        if len(waveform) != 10:
            a = 10

        label = self.labels_df[idx]

        return waveform, label

labels_df = pd.read_excel('./data/NEW_IRB300012145_Patient_ID_deidentified.xlsx')
Smokeing_status = labels_df.iloc[:, 4].to_list()

transform = AmplitudeNormalization()
data_dir = './data/Patients'
sound_dataset = SoundDataset(data_dir, Smokeing_status, transform=transform)

dataloader = DataLoader(sound_dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    waveforms, labels = batch
    print(waveforms.shape)
    print(labels)