"""
Dataset utilities for loading audio and lyrics features.
Provide functions to extract features (MFCC, spectrogram) and text embeddings.
"""

import os
import numpy as np
import librosa


def load_audio_file(path, sr=22050):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def extract_mfcc(y, sr=22050, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


class MusicDataset:
    def __init__(self, audio_dir, lyrics_dir=None):
        self.audio_dir = audio_dir
        self.lyrics_dir = lyrics_dir
        self.items = self._scan()

    def _scan(self):
        files = []
        for fname in os.listdir(self.audio_dir):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                files.append(fname)
        return files

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname = self.items[idx]
        path = os.path.join(self.audio_dir, fname)
        y = load_audio_file(path)
        mfcc = extract_mfcc(y)
        return fname, mfcc
