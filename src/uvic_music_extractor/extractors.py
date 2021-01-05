#!/usr/bin/env python

"""
Audio Feature Extractors
"""

from abc import ABC, abstractmethod
import numpy as np
import essentia
import essentia.standard as es


class ExtractorBase(ABC):
    """
    Base class for audio feature extractors

    :param sample_rate (int): rate to run extractors at
    :param stats (list): stats to run during aggregation (if used)
    """

    def __init__(self, sample_rate: float, stats=["mean", "stdev"]):
        self.sample_rate = sample_rate
        self.stats = stats.copy()

    @abstractmethod
    def __call__(self, audio: np.ndarray):
        """
        Abstract method -- must be implemented in inheriting classes

        :param audio (np.ndarray): input audio to run feature extraction on
        :return:
        """
        pass


class Spectral(ExtractorBase):
    """
    Spectral audio feature extraction
    """

    @staticmethod
    def get_feature_names():
        """
        Get a list of the features
        :return: list
        """

        return ["harsh", "energyLF", "rolloff_85", "rolloff_95", "spectral_centroid", "spectral_spread",
                "spectral_skewness", "spectral_kurtosis"]

    def get_headers(self, join="."):
        """
        Get a list of the features combined with aggregation
        :return: list
        """

        headers = []
        for feature in Spectral.get_feature_names():
            for stat in self.stats:
                headers.append("{}{}{}".format(feature, join, stat))

        return headers

    def __call__(self, audio: np.ndarray):
        """
        Run audio
        :param audio (np.ndarray): input audio
        :return:
        """

        pool = essentia.Pool()
        poolAgg = es.PoolAggregator(defaultStats=['mean', 'stdev'])
        window = es.Windowing()
        spectrum = es.Spectrum()
        centroid = es.Centroid(range=self.sample_rate/2)
        central_moments = es.CentralMoments(range=self.sample_rate/2)
        dist_shape = es.DistributionShape()

        energy_band_harsh = es.EnergyBandRatio(sampleRate=self.sample_rate, startFrequency=2000, stopFrequency=5000)
        energy_band_low = es.EnergyBandRatio(sampleRate=self.sample_rate, startFrequency=20, stopFrequency=80)
        rolloff_85 = es.RollOff(cutoff=0.85, sampleRate=self.sample_rate)
        rolloff_95 = es.RollOff(cutoff=0.95, sampleRate=self.sample_rate)

        #octaveFB = es.FrequencyBands(bands=[0,50,100,200,400,800,1600,3200,6400,12800])

        for frame in es.FrameGenerator(audio, 2048, 1024):
            win = window(frame)
            spec = spectrum(win)
            sc = centroid(spec)

            moments = central_moments(spec)
            spread, skewness, kurtosis = dist_shape(moments)

            harsh = energy_band_harsh(spec)
            energy_lf = energy_band_low(spec)
            roll85 = rolloff_85(spec)
            roll95 = rolloff_95(spec)

            keys = Spectral.get_feature_names()
            pool.add(keys[0], harsh)
            pool.add(keys[1], energy_lf)
            pool.add(keys[2], roll85)
            pool.add(keys[3], roll95)
            pool.add(keys[4], sc)
            pool.add(keys[5], spread)
            pool.add(keys[6], skewness)
            pool.add(keys[7], kurtosis)

        stats = poolAgg(pool)
        results = [stats[feature] for feature in self.get_headers()]
        return results
