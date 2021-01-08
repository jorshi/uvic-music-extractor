#!/usr/bin/env python

"""
Utility functions

Jordie Shier
University of Victoria
"""

import os
import numpy as np
import essentia.standard as es


def get_audio_files(location, sort=True):
    """
    Search the location provided for audio files

    :param location: (str) - path of audio file or directory of files
    :param sort: (bool) - return the list of audio files in sorted order, defaults to True
    :return: (list) - audio files
    """

    # Get list of audio samples - either from a directory or single file
    audio_files = []
    if os.path.isdir(location):
        audio_files = [os.path.abspath(os.path.join(location, f)) for f in os.listdir(location) if f.endswith('.wav')]

    elif os.path.isfile(location) and location.endswith('.wav'):
        audio_files = [os.path.abspath(location)]

    if not audio_files:
        raise RuntimeError("Could not find any audio files at location: {}".format(location))

    if sort:
        audio_files = sorted(audio_files)

    return audio_files


def load_audio(path, sample_rate, mono=True):
    """
    Load an audio file using Essentia

    :param path: (str) location of audio file to load
    :param sample_rate: (int) sampling rate to load audio at
    :param mono: (bool) convert file to mono, defaults to True
    :return: audio samples
    """

    # Load audio file
    loader = es.AudioLoader(filename=path)
    results = loader()
    samples = results[0]
    orig_rate = results[1]
    channels = results[2]

    # Make sure we get a mono or stereo audio
    if channels > 2:
        raise RuntimeError("Can't handle more than two audio channels.")

    # If there is only one channel, duplicate the first over to the second.
    # Essentia always loads as a stereo audio file and the right channel is
    # all zeros in this case. We'll convert to a stereo file for some of the
    # processing here such as the Loudness Normalization.
    if channels == 1:
        samples[:, 1] = samples[:, 0]

    # Mix to mono if required
    if mono:
        samples = mix_to_mono(samples)

    # Perform resampling if required
    if orig_rate != sample_rate:
        resample = es.Resample(inputSampleRate=orig_rate, outputSampleRate=sample_rate)

        # Resampling for a stereo audio file
        if not mono:
            resampled_left = resample(samples[:, 0])
            resampled_right = resample(samples[:, 1])
            samples = np.array([resampled_left, resampled_right])
            samples = samples.T

        # Resampling for a mono audio file
        else:
            samples = resample(samples)

    return samples, channels


def mix_to_mono(audio):
    """
    Mix an audio file down to mono

    :param audio: (np.ndarray) audio samples
    :return: (nd.ndarray) mono audio samples
    """

    mono_mix = es.MonoMixer()
    samples = mono_mix(audio, audio.shape[1])
    return samples


def normalize_loudness(audio, sample_rate, lufs=-24):
    """
    Normalize input audio to a specified value in LUFS

    :param audio: (np.ndarray) audio samples
    :param sample_rate: (int) sample rate
    :param lufs: (float) loudness goal in LUFS
    :return: (np.ndarray) normalized audio samples
    """

    # Get the current loudness in LUFS
    loudness = es.LoudnessEBUR128(startAtZero=True, sampleRate=sample_rate)
    results = loudness(audio)
    current_lufs = results[2]

    # Amount in dB that the file needs to be adjusted
    adjustment = lufs - current_lufs

    # Apply adjustment to the audio file
    gain = pow(10, adjustment / 20)
    normalized = audio * gain

    return normalized


def rms(audio: np.ndarray) -> float:
    """
    Calculate the RMS level for an array

    :param audio: input audio
    :return: (float) rms
    """
    result = np.mean(audio * audio)
    if result != 0.0:
        result = np.sqrt(result)

    return result
