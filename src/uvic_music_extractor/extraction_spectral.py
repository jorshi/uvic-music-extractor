#!/usr/bin/env python

"""
Spectral Feature extraction script for Lethbridge Mix Study
"""

from __future__ import print_function
import csv
import os
import sys
import math
import argparse
import numpy as np
import essentia
import essentia.standard as es
import librosa
from scipy.stats import norm

SR = 48000

def runExtraction(location):

    # Get list of audio samples - either from a directory or single file
    audioFiles = []
    if (os.path.isdir(location)):
        audioFiles = [os.path.abspath(os.path.join(location,f)) for f in os.listdir(location) if f.endswith('.wav')]
    elif (os.path.isfile(location) and location.endswith('.wav')):
        audioFiles = [os.path.abspath(location)]
    if not audioFiles:
        print("No audio files found!")
        sys.exit(1)

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




def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', help="Input directory", type=str)
    parser.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('w'))

    args = parser.parse_args(arguments)

    results = runExtraction(args.input)

    wr = csv.writer(args.outfile, quoting=csv.QUOTE_ALL)
    for row in results:
        wr.writerow(row)

    #np.savetxt(args.outfile, results, fmt="%s", delimiter=',')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
