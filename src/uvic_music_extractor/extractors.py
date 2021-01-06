#!/usr/bin/env python

"""
Audio Feature Extractors
"""

from abc import ABC, abstractmethod
import numpy as np
import essentia
import essentia.standard as es
import librosa


class ExtractorBase(ABC):
    """
    Base class for audio feature extractors

    :param sample_rate (int): rate to run extractors at
    :param stats (list): stats to run during pooling aggregation (if used)
    """

    def __init__(self, sample_rate: float, pooling: bool = False, stats: list = None):
        self.sample_rate = sample_rate
        self.pooling = pooling
        if stats is None:
            self.stats = ["mean", "stdev"]

    @abstractmethod
    def __call__(self, audio: np.ndarray):
        """
        Abstract method -- must be implemented in inheriting classes

        :param audio (np.ndarray): input audio to run feature extraction on
        :return:
        """
        pass

    def get_headers(self, join="."):
        """
        Get a list of the features combined with aggregation
        :return: list
        """

        if not self.pooling:
            return self.__class__.get_feature_names()

        headers = []
        for feature in self.__class__.get_feature_names():
            for stat in self.stats:
                headers.append("{}{}{}".format(feature, join, stat))

        return headers

    @staticmethod
    @abstractmethod
    def get_feature_names():
        """
        Get a list of the features
        :return: list
        """
        pass


class Spectral(ExtractorBase):
    """
    Spectral audio feature extraction
    """

    def __init__(self, sample_rate: float, stats: list = None):
        super().__init__(sample_rate, pooling=True, stats=stats)

    def __call__(self, audio: np.ndarray):
        """
        Run audio
        :param audio (np.ndarray): input audio
        :return: feature matrix
        """

        pool = essentia.Pool()
        pool_agg = es.PoolAggregator(defaultStats=self.stats)
        window = es.Windowing(type="hann", size=2048)
        spectrum = es.Spectrum()

        # Spectral Features
        centroid = es.Centroid(range=self.sample_rate/2)
        central_moments = es.CentralMoments(range=self.sample_rate/2)
        dist_shape = es.DistributionShape()
        flatness = es.Flatness()
        entropy = es.Entropy()

        energy_band_harsh = es.EnergyBandRatio(sampleRate=self.sample_rate, startFrequency=2000, stopFrequency=5000)
        energy_band_low = es.EnergyBandRatio(sampleRate=self.sample_rate, startFrequency=20, stopFrequency=80)
        rolloff_85 = es.RollOff(cutoff=0.85, sampleRate=self.sample_rate)
        rolloff_95 = es.RollOff(cutoff=0.95, sampleRate=self.sample_rate)

        for frame in es.FrameGenerator(audio, 2048, 1024):

            win = window(frame)
            spec = spectrum(win)

            # Spectral Centroid
            sc = centroid(spec)
            moments = central_moments(spec)
            spread, skewness, kurtosis = dist_shape(moments)
            spectral_flatness = flatness(spec)
            spectral_entropy = entropy(spec)

            harsh = energy_band_harsh(spec)
            energy_lf = energy_band_low(spec)
            roll85 = rolloff_85(spec)
            roll95 = rolloff_95(spec)

            keys = Spectral.get_feature_names()
            pool.add(keys[0], roll85)
            pool.add(keys[1], roll95)
            pool.add(keys[2], sc)
            pool.add(keys[3], spread)
            pool.add(keys[4], skewness)
            pool.add(keys[5], kurtosis)
            pool.add(keys[6], spectral_flatness)
            pool.add(keys[7], spectral_entropy)
            pool.add(keys[8], harsh)
            pool.add(keys[9], energy_lf)

        stats = pool_agg(pool)
        results = [stats[feature] for feature in self.get_headers()]
        return results

    @staticmethod
    def get_feature_names():
        """
        Get a list of the features
        :return: list
        """

        return ["rolloff_85", "rolloff_95", "spectral_centroid", "spectral_spread",
                "spectral_skewness", "spectral_kurtosis", "spectral_flatness", "spectral_entropy",
                "harsh", "energyLF"]


class CrestFactor(ExtractorBase):
    """
    Crest Factor Extractor
    """

    def __init__(self, sample_rate: float, frame_size: float = None, stats: list = None):
        super().__init__(sample_rate, pooling=frame_size is not None, stats=stats)
        self.frame_size = frame_size

    def __call__(self, audio: np.ndarray):
        """
        Run crest factor audio feature extraction

        :param audio: Input audio samples
        :return: feature matrix
        """

        rms = es.RMS()
        minimum = es.MinMax(type='min')
        maximum = es.MinMax(type='max')

        if self.frame_size:
            pool = essentia.Pool()
            pool_agg = es.PoolAggregator(defaultStats=self.stats)
            for frame in es.FrameGenerator(audio, self.frame_size, self.frame_size / 2):
                frame_rms = rms(frame)
                frame_peak_min = minimum(frame)[0]
                frame_peak_max = maximum(frame)[0]
                frame_peak = max(abs(frame_peak_min), abs(frame_peak_max))
                frame_crest = frame_peak / frame_rms
                pool.add('crest_factor', frame_crest)

            stats = pool_agg(pool)
            crest_factor = [stats['crest_factor.{}'.format(stat)] for stat in self.stats]

        else:
            full_rms = rms(audio)
            full_peak_min = minimum(audio)[0]
            full_peak_max = maximum(audio)[0]
            full_peak = max(abs(full_peak_min), abs(full_peak_max))
            crest_factor = full_peak / full_rms

        return crest_factor

    @staticmethod
    def get_feature_names():
        """
        Get a list of the features
        :return: list
        """
        return ["crest_factor"]
