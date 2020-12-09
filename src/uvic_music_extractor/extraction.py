#!/usr/bin/env python

"""
Feature extraction script for Lethbridge Mix Study
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
from scipy.stats import norm

SR = 48000

def crestFactor(sample, frameSize=None):

    # Get audio from file
    loader = es.MonoLoader(filename=sample, sampleRate=SR)
    audio = loader()

    rms = es.RMS()
    minimum = es.MinMax(type='min')
    maximum = es.MinMax(type='max')
    crestFactor = 0.0

    if frameSize:
        pool = essentia.Pool()
        poolAgg = es.PoolAggregator(defaultStats=['mean', 'stdev'])
        for frame in es.FrameGenerator(audio, frameSize, frameSize):
            frameRMS = rms(frame)
            framePeakMin = minimum(frame)[0]
            framePeakMax = maximum(frame)[0]
            framePeak = max(abs(framePeakMin), abs(framePeakMax))
            frameCrest = framePeak / frameRMS
            pool.add('crestFactor', frameCrest)

        stats = poolAgg(pool)
        crestFactor = (stats['crestFactor.mean'], stats['crestFactor.stdev'])

    else:
        fullRMS = rms(audio)
        fullPeakMin = minimum(audio)[0]
        fullPeakMax = maximum(audio)[0]
        fullPeak = max(abs(fullPeakMin), abs(fullPeakMax))
        crestFactor = fullPeak / fullRMS

    return crestFactor


def loudnessFeatures(sample):

    loader = es.AudioLoader(filename=sample)
    audio = loader()

    loudness = es.LoudnessEBUR128(startAtZero=True, sampleRate=SR)
    loudnessStats = loudness(audio[0])
    loudnessRange = loudnessStats[3]

    # Microdynamics (LDR)
    microdynamics = loudnessStats[0] - loudnessStats[1]
    ldr95 = np.percentile(microdynamics, 95.0)
    ldrmax = microdynamics.max()

    # True peak detection for peak to loudness calculation
    truePeakDetector = es.TruePeakDetector(sampleRate=SR)
    truePeakAudioL = truePeakDetector(audio[0][:,0])[1]
    truePeakL = 20 * math.log10(truePeakAudioL.max())
    truePeakAudioR = truePeakDetector(audio[0][:,1])[1]
    truePeakR = 20 * math.log10(truePeakAudioR.max())

    truePeak = max(truePeakL, truePeakR)
    peakToLoudness = truePeak / loudnessStats[2]

    # top1db (probably won't be any??)
    top1dbGain = math.pow(10, -1.0/20.0)
    top1dbL = (truePeakAudioL > top1dbGain).sum()
    top1dbR = (truePeakAudioR > top1dbGain).sum()
    top1db = (top1dbL + top1dbR) / (len(truePeakAudioL) + len(truePeakAudioR))

    return loudnessRange, ldr95, ldrmax, peakToLoudness, top1db


def dynamicSpread(sample, frameSize=2048):

    loader = es.MonoLoader(filename=sample)
    audio = loader()

    vickersLoudness = es.LoudnessVickers()
    pool = essentia.Pool()
    poolAgg = es.PoolAggregator(defaultStats=['mean'])

    for frame in es.FrameGenerator(audio, frameSize, frameSize):
        frameLoud = vickersLoudness(frame)
        pool.add('vdb', frameLoud)

    stats = poolAgg(pool)
    vmean = stats['vdb.mean']

    dynamicSpread = 0.0
    for vdb in pool['vdb']:
        dynamicSpread += abs(vdb - vmean)

    dynamicSpread /= len(pool['vdb'])

    return dynamicSpread


def distortion(sample):

    loader = es.MonoLoader(filename=sample, sampleRate=SR)
    audio = loader()

    hist, edges = np.histogram(audio, 1001, (-1.0,1.0))
    hist = np.array(hist, dtype=np.float32)

    centroidCalc = es.Centroid()
    centroid = centroidCalc(hist)

    centralMoments = es.CentralMoments()
    shape = es.DistributionShape()

    cm = centralMoments(hist)
    spread, skewness, kurtosis = shape(cm)

    flatnessCalc = es.Flatness()
    flatness = flatnessCalc(hist)

    prime = np.zeros(1000)
    for i in range(1, 1000):
        dy = abs(hist[i] - hist[i-1])
        prime[i-1] = dy

    domain = np.linspace(-1.0, 1.0, 1000)
    gaussHist = norm.pdf(domain,0.0,0.2)

    correlation_matrix = np.corrcoef(prime, gaussHist)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2

    return centroid, spread, skewness, kurtosis, flatness, r_squared



def stereoFeatures(sample):

    loader = es.AudioLoader(filename=sample)
    audio = loader()[0]

    sides = (audio[:,0] - audio[:,1]) ** 2
    mids = (audio[:,0] + audio[:,1]) ** 2

    sides_mid_ratio = sides.mean() / mids.mean()

    leftPower = (audio[:,0] ** 2).mean()
    rightPower = (audio[:,1] ** 2).mean()

    lr_imbalance = (rightPower - leftPower) / (rightPower + leftPower)

    return sides_mid_ratio, lr_imbalance



def phaseCorrelation(sample, frame_size=None):

    loader = es.AudioLoader(filename=sample)
    audio = loader()[0]

    phase_correlation = 1.0

    if frame_size:
        max_sample = audio.shape[0]
        slice_indices = list(range(0, max_sample, frame_size))
        slice_indices.append(max_sample)

        pool = essentia.Pool()
        for i in range(len(slice_indices) - 1):
            x1 = slice_indices[i]
            x2 = slice_indices[i + 1]
            correlation_matrix = np.corrcoef(audio[x1:x2,0], audio[x1:x2,1])
            phase_correlation = correlation_matrix[0,1]
            pool.add('phase_correlation', phase_correlation)


        poolAgg = es.PoolAggregator(defaultStats=['mean', 'stdev'])
        stats = poolAgg(pool)
        phase_correlation = (stats['phase_correlation.mean'], stats['phase_correlation.stdev'])

    else:
        correlation_matrix = np.corrcoef(audio[:,0], audio[:,1])
        phase_correlation = correlation_matrix[0,1]


    return phase_correlation



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
        "phase_correlation_full",
        "phase_correlation_1s_mean",
        "phase_correlation_1s_std",
        "phase_correlation_100ms_mean",
        "phase_correlation_100ms_std",
        "sides_to_mid_ratio",
        "left_right_imbalance",
        "pmf_centroid",
        "pmf_spread",
        "pmf_skew",
        "pmf_kurtosis",
        "pmf_flatness",
        "pmf_gauss",
        "loudness_range",
        "microdynamics_95%",
        "microdynamics_100%",
        "peak_to_loudness",
        "top1db",
        "dynamic_spread",
        "crest_factor_full",
        "crest_factor_1s_mean",
        "crest_factor_1s_std",
        "crest_factor_100ms_mean",
        "crest_factor_100ms_std"
    ]]

    # Run feature extraction on all audio files
    for audio in sorted(audioFiles):

        results = [audio]

        print('Running feature extraction for %s' % audio)
        print('Phase Correlation')
        phase_corr_full = phaseCorrelation(audio)
        phase_corr_1s = phaseCorrelation(audio, 44100)
        phase_corr_100ms = phaseCorrelation(audio, 4410)
        results.extend([
            phase_corr_full,
            phase_corr_1s[0],
            phase_corr_1s[1],
            phase_corr_100ms[0],
            phase_corr_100ms[1],
        ])

        print('Stereo Width')
        sides_mid_ratio, lr_imbalance = stereoFeatures(audio)
        results.extend([sides_mid_ratio, lr_imbalance])

        print('Distortion')
        distotion_features = distortion(audio)
        pmf_centroid = distotion_features[0]
        pmf_spread = distotion_features[1]
        pmf_skewness = distotion_features[2]
        pmf_kurtosis = distotion_features[3]
        pmf_flatness = distotion_features[4]
        gauss = distotion_features[5]
        results.extend(distotion_features)

        print('Loudness and Dynamics')
        lra, ldr95, ldrmax, peak_to_loudness, top1db = loudnessFeatures(audio)
        dynamic_spread = dynamicSpread(audio)
        results.extend([lra, ldr95, ldrmax, peak_to_loudness, top1db, dynamic_spread])

        print('Crest Factor')
        cf_full = crestFactor(audio)
        cf_1s = crestFactor(audio, 44100)
        cf_100ms = crestFactor(audio, 4410)
        results.extend([
            cf_full,
            cf_1s[0],
            cf_1s[1],
            cf_100ms[0],
            cf_100ms[1],
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
