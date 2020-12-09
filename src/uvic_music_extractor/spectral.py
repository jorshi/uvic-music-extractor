#!/usr/bin/env python

"""
Spectral Feature Extractor

Jordie Shier
University of Victoria
"""

import os
import sys
import essentia
import essentia.standard as es

SR = 48000

def runExtraction(audio, sample_rate):


    output = [[
        "file",
        "harsh_mean",
        "harsh_std",
        "lf_energy_mean",
        "lf_energy_std",
        "rolloff_85_mean",
        "rolloff_85_std",
        "rolloff_95_mean",
        "rolloff_95_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_spread_mean",
        "spectral_spread_std",
        "spectral_skewness_mean",
        "spectral_skewness_std",
        "spectral_kurtosis_mean",
        "spectral_kurtosis_std",
    ]]

    # Run feature extraction on all audio files
    for audio in sorted(audioFiles):

        results = [audio]
        print('Running feature extraction for %s' % audio)

        loader = es.MonoLoader(filename=audio, sampleRate=SR)
        samples = loader()

        pool = essentia.Pool()
        poolAgg = es.PoolAggregator(defaultStats=['mean', 'stdev'])
        window = es.Windowing()
        spectrum = es.Spectrum()
        centroid = es.Centroid(range=SR/2)
        central_moments = es.CentralMoments(range=SR/2)
        dist_shape = es.DistributionShape()

        energyBandHarsh = es.EnergyBandRatio(sampleRate=SR, startFrequency=2000, stopFrequency=5000)
        energyBandLF = es.EnergyBandRatio(sampleRate=SR, startFrequency=20, stopFrequency=80)
        rolloff85 = es.RollOff(cutoff=0.85, sampleRate=SR)
        rolloff95 = es.RollOff(cutoff=0.95, sampleRate=SR)

        #octaveFB = es.FrequencyBands(bands=[0,50,100,200,400,800,1600,3200,6400,12800])

        for frame in es.FrameGenerator(samples, 2048, 1024):
            win = window(frame)
            spec = spectrum(win)
            sc = centroid(spec)

            moments = central_moments(spec)
            spread, skewness, kurtosis = dist_shape(moments)

            harsh = energyBandHarsh(spec)
            energyLF = energyBandLF(spec)
            roll85 = rolloff85(spec)
            roll95 = rolloff95(spec)
            pool.add('harsh', harsh)
            pool.add('energyLF', energyLF)
            pool.add('rolloff_85', roll85)
            pool.add('rolloff_95', roll95)
            pool.add('sc', sc)
            pool.add('spread', spread)
            pool.add('skewness', skewness)
            pool.add('kurtosis', kurtosis)

        stats = poolAgg(pool)
        results.extend([stats['harsh.mean'],
                        stats['harsh.stdev'],
                        stats['energyLF.mean'],
                        stats['energyLF.stdev'],
                        stats['rolloff_85.mean'],
                        stats['rolloff_85.stdev'],
                        stats['rolloff_95.mean'],
                        stats['rolloff_95.stdev'],
                        stats['sc.mean'],
                        stats['sc.stdev'],
                        stats['spread.mean'],
                        stats['spread.stdev'],
                        stats['skewness.mean'],
                        stats['skewness.stdev'],
                        stats['kurtosis.mean'],
                        stats['kurtosis.stdev']
                        ])


        output.append(results)

    return output