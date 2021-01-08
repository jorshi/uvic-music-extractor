#!/usr/bin/env python

"""
Audio Feature Extractors
"""

from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.stats import norm
import essentia
import essentia.standard as es

import uvic_music_extractor.utils as utils


class ExtractorBase(ABC):
    """
    Base class for audio feature extractors

    :param sample_rate (int): rate to run extractors at
    :param stats (list): stats to run during pooling aggregation (if used)
    """

    def __init__(self, sample_rate: float, pooling: bool = False, stats: list = None):
        self.sample_rate = sample_rate
        self.pooling = pooling
        self.feature_names = []
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
            return self.feature_names

        headers = []
        for feature in self.feature_names:
            for stat in self.stats:
                headers.append("{}{}{}".format(feature, join, stat))

        return headers


class Spectral(ExtractorBase):
    """
    Spectral audio feature extraction
    """

    def __init__(self, sample_rate: float, stats: list = None):
        super().__init__(sample_rate, pooling=True, stats=stats)
        self.feature_names = ["rolloff_85", "rolloff_95", "spectral_centroid", "spectral_spread",
                "spectral_skewness", "spectral_kurtosis", "spectral_flatness", "spectral_entropy",
                "harsh", "energyLF"]

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

            keys = self.feature_names
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


class CrestFactor(ExtractorBase):
    """
    Crest Factor Extractor
    """

    def __init__(self, sample_rate: float, frame_size: float = None, stats: list = None):
        super().__init__(sample_rate, pooling=frame_size is not None, stats=stats)
        self.frame_size = frame_size
        self.feature_names = ["crest_factor"]

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
            for frame in es.FrameGenerator(audio, self.frame_size, self.frame_size):
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
            crest_factor = [full_peak / full_rms]

        return crest_factor


class Loudness(ExtractorBase):
    """
    Loudness Features
    """

    def __init__(self, sample_rate: float, stats: list = None):
        super().__init__(sample_rate, pooling=False, stats=stats)
        self.feature_names = ["loudness_range", "microdynamics_95%", "microdynamics_100%",
                              "peak_to_loudness", "top1db"]

    def __call__(self, audio: np.ndarray):
        """
        Run loudness feature extraction

        :param audio: Input audio samples
        :return: feature matrix
        """

        loudness = es.LoudnessEBUR128(startAtZero=True, sampleRate=self.sample_rate)
        loudness_stats = loudness(audio)
        loudness_range = loudness_stats[3]

        # Micro dynamics (LDR)
        micro_dynamics = loudness_stats[0] - loudness_stats[1]
        ldr_95 = np.percentile(micro_dynamics, 95.0)
        ldr_max = micro_dynamics.max()

        # True peak detection for peak to loudness calculation
        true_peak_detector = es.TruePeakDetector(sampleRate=self.sample_rate)
        true_peak_audio_l = true_peak_detector(audio[:, 0])[1]
        true_peak_l = 20 * math.log10(true_peak_audio_l.max())
        true_peak_audio_r = true_peak_detector(audio[:, 1])[1]
        true_peak_r = 20 * math.log10(true_peak_audio_r.max())

        # True peak to loudness
        true_peak = max(true_peak_l, true_peak_r)
        peak_to_loudness = true_peak / loudness_stats[2]

        # Top 1 dB (ratio of samples in the top 1dB)
        top_1db_gain = math.pow(10, -1.0 / 20.0)
        top_1db_l = (true_peak_audio_l > top_1db_gain).sum()
        top_1db_r = (true_peak_audio_l > top_1db_gain).sum()
        top1db = (top_1db_l + top_1db_r) / (len(true_peak_audio_l) + len(true_peak_audio_r))

        return [loudness_range, ldr_95, ldr_max, peak_to_loudness, top1db]


class DynamicSpread(ExtractorBase):
    """
    Dynamic Spread Feature Extractor
    """

    def __init__(self, sample_rate: float, frame_size: float = 2048, stats: list = None):
        super().__init__(sample_rate, pooling=False, stats=stats)
        self.frame_size = frame_size
        self.feature_names = ["dynamic_spread"]

    def __call__(self, audio: np.ndarray):
        """
        Run loudness feature extraction

        :param audio: Input audio samples
        :return: feature matrix
        """

        vickers_loudness = es.LoudnessVickers()
        pool = essentia.Pool()
        pool_agg = es.PoolAggregator(defaultStats=['mean'])

        # Calculate the Vickers loudness frame by frame
        for frame in es.FrameGenerator(audio, self.frame_size, self.frame_size):
            frame_loudness = vickers_loudness(frame)
            pool.add('vdb', frame_loudness)

        # Compute the average loudness across frames
        stats = pool_agg(pool)
        vickers_mean = stats['vdb.mean']

        # Compute the difference between loudness at each frame and the mean loudness
        dynamic_spread = 0.0
        for vdb in pool['vdb']:
            dynamic_spread += abs(vdb - vickers_mean)

        dynamic_spread /= len(pool['vdb'])

        return [dynamic_spread]


class Distortion(ExtractorBase):
    """
    Set of distortion features
    """

    def __init__(self, sample_rate: float, stats: list = None):
        super().__init__(sample_rate, pooling=False, stats=stats)
        self.feature_names = ["pmf_centroid", "pmf_spread", "pmf_skewness",
                              "pmf_kurtosis", "pmf_flatness", "pmf_gauss"]

    def __call__(self, audio: np.ndarray):
        """
        Run distortion feature extraction

        :param audio: Input audio samples
        :return: feature matrix
        """

        hist, edges = np.histogram(audio, 1001, (-1.0, 1.0))
        hist = np.array(hist, dtype=np.float32)

        centroid_calc = es.Centroid()
        centroid = centroid_calc(hist)

        central_moments = es.CentralMoments()
        shape = es.DistributionShape()

        cm = central_moments(hist)
        spread, skewness, kurtosis = shape(cm)

        flatness_calc = es.Flatness()
        flatness = flatness_calc(hist)

        prime = np.zeros(1000)
        for i in range(1, 1000):
            dy = abs(hist[i] - hist[i - 1])
            prime[i - 1] = dy

        domain = np.linspace(-1.0, 1.0, 1000)
        gauss_hist = norm.pdf(domain, 0.0, 0.2)

        correlation_matrix = np.corrcoef(prime, gauss_hist)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2

        return [centroid, spread, skewness, kurtosis, flatness, r_squared]


class StereoFeatures(ExtractorBase):
    """
    Stereo Feature Extractor
    """

    def __init__(self, sample_rate: float, stats: list = None):
        super().__init__(sample_rate, pooling=False, stats=stats)
        self.feature_names = ["side_mid_ratio", "lr_imbalance"]

    def __call__(self, audio: np.ndarray):
        """
        Run stereo feature extraction

        :param audio: Input audio samples
        :return: feature matrix
        """

        sides = (audio[:, 0] - audio[:, 1]) ** 2
        mids = (audio[:, 0] + audio[:, 1]) ** 2

        sides_mid_ratio = sides.mean() / mids.mean()

        left_power = (audio[:, 0] ** 2).mean()
        right_power = (audio[:, 1] ** 2).mean()

        lr_imbalance = (right_power - left_power) / (right_power + left_power)

        return sides_mid_ratio, lr_imbalance


class PhaseCorrelation(ExtractorBase):
    """
    Phase Correlation Features
    """

    def __init__(self, sample_rate: float, frame_size: float = None, stats: list = None):
        super().__init__(sample_rate, pooling=frame_size is not None, stats=stats)
        self.frame_size = frame_size
        self.feature_names = ["phase_correlation"]

    def __call__(self, audio: np.ndarray):
        """
        Run phase correlation feature extraction

        :param audio: Input audio samples
        :return: feature matrix
        """

        if self.frame_size:
            max_sample = audio.shape[0]
            slice_indices = list(range(0, max_sample, self.frame_size))
            slice_indices.append(max_sample)

            pool = essentia.Pool()
            for i in range(len(slice_indices) - 1):
                x1 = slice_indices[i]
                x2 = slice_indices[i + 1]
                correlation_matrix = np.corrcoef(audio[x1:x2, 0], audio[x1:x2, 1])
                phase_correlation = correlation_matrix[0, 1]
                pool.add(self.feature_names[0], phase_correlation)

            pool_agg = es.PoolAggregator(defaultStats=self.stats)
            stats = pool_agg(pool)
            phase_correlation = [stats["{}.{}".format(self.feature_names[0], stat)] for stat in self.stats]

        else:
            correlation_matrix = np.corrcoef(audio[:, 0], audio[:, 1])
            phase_correlation = [correlation_matrix[0, 1]]

        return phase_correlation


class StereoSpectrum(ExtractorBase):
    """
    Stereo Spectrum Features
    """

    def __init__(self, sample_rate: float, frame_size: int = 2048,
                 hop_size: int = 1024, stats: list = None):
        super().__init__(sample_rate, pooling=True, stats=stats)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.low = 250
        self.high = 2800
        self.feature_names = ["sps_full", "sps_low", "sps_mid", "sps_high"]

    def __call__(self, audio: np.ndarray):
        """
        Run stereo spectrum feature extraction

        :param audio: Input audio samples
        :return: feature matrix
        """

        # Must be stereo audio
        assert audio.shape[1] == 2

        # Hanning window
        window = np.hanning(self.frame_size)

        pool = essentia.Pool()
        pool_agg = es.PoolAggregator(defaultStats=self.stats)

        # Bin numbers for each filter bank
        low_bin = int((self.low / self.sample_rate) * self.frame_size)
        assert low_bin <= int(self.frame_size / 2)

        high_bin = int((self.high / self.sample_rate) * self.frame_size)
        assert high_bin <= int(self.frame_size / 2)

        for i in range(0, len(audio), self.hop_size):
            # Get the windowed frame for each channel
            samples = audio[i:i+self.frame_size, :]
            frame_left = np.zeros(self.frame_size)
            frame_left[:len(samples)] = samples[:, 0]
            frame_right = np.zeros(self.frame_size)
            frame_right[:len(samples)] = samples[:, 1]

            # Apply window
            frame_left *= window
            frame_right *= window

            X_left = np.fft.rfft(frame_left)
            X_right = np.fft.rfft(frame_right)

            stereo_spectrum = StereoSpectrum.compute_stereo_spectrum(X_left, X_right)

            # Features
            full = utils.rms(stereo_spectrum)
            low = utils.rms(stereo_spectrum[:low_bin])
            mid = utils.rms(stereo_spectrum[low_bin:high_bin])
            high = utils.rms(stereo_spectrum[high_bin:])

            pool.add(self.feature_names[0], full)
            pool.add(self.feature_names[1], low)
            pool.add(self.feature_names[2], mid)
            pool.add(self.feature_names[3], high)

        stats = pool_agg(pool)
        results = [stats[feature] for feature in self.get_headers()]

        return results

    @staticmethod
    def compute_stereo_spectrum(spectrum_left, spectrum_right):

        np.zeros_like(spectrum_left)

        # Update the DC and Nyquist Bins
        spectrum_left[0] = np.real(spectrum_left[0]) + 0j
        spectrum_left[-1] = np.real(spectrum_left[-1]) + 0j
        spectrum_right[0] = np.real(spectrum_right[0]) + 0j
        spectrum_right[-1] = np.real(spectrum_right[-1]) + 0j

        real_left = np.real(spectrum_left)
        imag_left = np.imag(spectrum_left)
        real_right = np.real(spectrum_right)
        imag_right = np.imag(spectrum_right)

        f1 = (real_left * real_right) * (real_left * real_right)
        f2 = (imag_left * imag_right) * (imag_left * imag_right)
        f3 = (imag_left * real_right) * (imag_left * real_right)
        f4 = (imag_right * real_left) * (imag_right * real_left)
        nf = np.sqrt(f1 + f2 + f3 + f4)
        dfl = real_left * real_left + imag_left * imag_left
        dfr = real_right * real_right + imag_right * imag_right
        df = dfl + dfr

        sign = nf / dfl - nf / dfr
        sign[sign > 0] = 1.0
        sign[sign < 0] = -1.0

        return (1.0 - 2.0 * (nf / df)) * sign


class SpectralFlux(ExtractorBase):
    """
    Spectral Flux Features
    """

    def __init__(self, sample_rate: float, frame_size: float = 2048,
                 num_bands: int = 10, stats: list = None):
        super().__init__(sample_rate, pooling=True, stats=stats)
        self.frame_size = frame_size
        self.num_bands = num_bands
        self.band_str = "spectral_flux_band_{}"
        self.feature_names = [self.band_str.format(i + 1) for i in range(self.num_bands)]

    def __call__(self, audio: np.ndarray):
        """
        Run spectral flux calculations

        :param audio: Input audio samples
        :return: feature matrix
        """

        pool = essentia.Pool()
        pool_agg = es.PoolAggregator(defaultStats=self.stats)

        window = es.Windowing(type="hann", size=self.frame_size)
        spectrum = es.Spectrum()

        # Initialize a Flux algorithm for each band
        sub_band_flux = [es.Flux() for i in range(self.num_bands)]

        # Calculate octave bands
        bands = [0]
        bands.extend([50 * 2**i for i in range(self.num_bands - 1)])

        # The top band must be less than the Nyquist
        assert bands[-1] < self.sample_rate / 2
        bands.append(self.sample_rate / 2)

        # Bands in FFT bins
        bands = np.round((np.array(bands) / self.sample_rate) * self.frame_size)

        # Run frame-by-frame processing with a one half hop size
        for frame in es.FrameGenerator(audio, self.frame_size, self.frame_size // 2):
            win = window(frame)
            spec = spectrum(win)

            # Calculate flux for each sub band
            for i in range(1, len(bands)):
                sub_band = spec[int(bands[i-1]):int(bands[i])]
                flux = sub_band_flux[i-1](sub_band)
                pool.add(self.band_str.format(i), flux)

        stats = pool_agg(pool)
        results = [stats[feature] for feature in self.get_headers()]
        return results
