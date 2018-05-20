#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Misc functions."""
import os

import numpy as np
import scipy.signal
import scipy.io.wavfile as wav
from python_speech_features import mfcc, delta, fbank


# Generate PCEN (per-channel energy normalized) Mel-spectrograms.
# https://arxiv.org/pdf/1607.05666.pdf
# https://github.com/librosa/librosa/issues/615
def gen_pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):
    M = scipy.signal.lfilter([s], [1, s - 1], E)
    smooth = (eps + M)**(-alpha)
    return (E * smooth + delta)**r - delta**r


# Generate pcen features from all .wav files in a data folder.
def gen_features(wvdir):
    if not os.path.isdir(wvdir):
        return np.array([])
    wvfiles = os.listdir(wvdir)
    # wav files only.
    wvfiles = list(filter(lambda f: f.endswith(".wav"), wvfiles))
    # generate full dir path for each wvfile.
    wvfiles = list(map(lambda f: os.path.join(wvdir, f), wvfiles))
    pcen_features = []
    for wvfile in wvfiles:
        pcen_features.append(gen_feature(wvfile))
    return np.array(pcen_features)


# Generate pcen feature from a .wav file.
# https://github.com/jameslyons/python_speech_features
def gen_feature(wvfile):
    (rate, sig) = wav.read(wvfile)
    # For multi-channel .wav file, use the first channel.
    if len(sig.shape) == 2:
        sig = sig[:, 0]
    # 40-channel Mel-filterbank, 25ms windowsing, 10ms frame shift.
    (fb_feature, total_energy) = fbank(sig, rate, nfft=2048, nfilt=40)
    pcen_feature = gen_pcen(fb_feature)
    return pcen_feature


# Transfer .wav files to features, split into train and test by 80:20,
# store features in .npy files
def prepare_data(posdir, negdir, destdir, ratio=0.8):
    # Clean destination directory if files already exist.
    if not os.path.isdir(destdir):
        os.mkdir(destdir)
    npyfiles = os.listdir(destdir)
    if "pos_train.npy" in npyfiles:
        os.remove(os.path.join(destdir, "pos_train.npy"))
    if "pos_test.npy" in npyfiles:
        os.remove(os.path.join(destdir, "pos_test.npy"))
    if "neg_train.npy" in npyfiles:
        os.remove(os.path.join(destdir, "neg_train.npy"))
    if "neg_test.npy" in npyfiles:
        os.remove(os.path.join(destdir, "neg_test.npy"))

    # Generate pcen features from wvfiles.
    pos_features = gen_features(posdir)
    neg_features = gen_features(negdir)

    # Randomly split data.
    np.random.seed(23)
    pos_shuffle_indices = np.random.permutation(np.arange(len(pos_features)))
    neg_shuffle_indices = np.random.permutation(np.arange(len(neg_features)))
    pos_shuffled = pos_features[pos_shuffle_indices]
    neg_shuffled = neg_features[neg_shuffle_indices]
    pos_split_index = int(float(len(pos_shuffled)) * ratio)
    neg_split_index = int(float(len(neg_shuffled)) * ratio)
    pos_train = pos_shuffled[:pos_split_index]
    pos_test = pos_shuffled[pos_split_index:]
    neg_train = neg_shuffled[:neg_split_index]
    neg_test = neg_shuffled[neg_split_index:]

    # Save features.
    np.save(os.path.join(destdir, "pos_train.npy"), pos_train)
    np.save(os.path.join(destdir, "pos_test.npy"), pos_test)
    np.save(os.path.join(destdir, "neg_train.npy"), neg_train)
    np.save(os.path.join(destdir, "neg_test.npy"), neg_test)


# Load data and generate labels.
def load_data_and_labels(pos_data_file, neg_data_file):
    """
    Load feature data and generate labels.
    """
    # Load data from files.
    pos_data = np.load(pos_data_file)
    neg_data = np.load(neg_data_file)
    x_data = np.concatenate([pos_data, neg_data], 0)
    # Generate labels.
    pos_labels = [[0, 1] for _ in pos_data]
    neg_labels = [[1, 0] for _ in neg_data]
    y = np.concatenate([pos_labels, neg_labels], 0)
    return [x_data, y]


# Generate batch iterator.
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generate a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch.
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
