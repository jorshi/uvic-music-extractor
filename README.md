# uvic-music-extractor
An audio feature extraction toolkit designed to provide a comprehensive selection of audio features for analyzing music. The choosen features were selected based on previous work in the field of intelligent music production and audio recording research [1-5]. While most these features are available in existing audio feature extraction libraries, they are spread out over several different libraries written in different programming languages. The goal of this tool is to provide access to a large selection of commonly used features in a single Python package with limited dependencies. Specifcally, this package is written using [Essentia](https://essentia.upf.edu/index.html).

## Installation

**Requirements:**
- Python 3.6 or greater
- [Essentia](https://essentia.upf.edu/index.html) -- see installation guide: https://essentia.upf.edu/installing.html. Unfortunately Essentia python bindings are not available on Windows at the time of writing.

**Installation:**

Once you have Python and Essentia installed, this git repo can be cloned and the package installed.

```
git clone https://github.com/jorshi/uvic-music-extractor.git
cd uvic-music-extractor
python -m pip install .
```

Make sure that you are calling pip with the same python instance that you installed Essentia to. If you are unsure, you can check by making sure you can import essentia (the following should not raise any errors)
```
python
>>> import essentia
>>> 
```

## Usage

The main way to use this tool is through the command line. After installation the following command can be invoked from the command line. It doesn't have to be called from the same folder as the source code (i.e. you can move to a directory that contains a folder/file for analysis and access the command). 

```
uvic_music_extractor input output [-h] [--rate] [--normalize] 
```

Positional arguments. These are required.

- `input`: A folder containing audio files to process or a single audio file to process.
- `output`: Location/name of the csv file to output.

Optional arguments:
- `--rate`: Sample rate to run extraction at. Input audio files will be resampled to this rate if required. Defaults to 44100Hz.
- `--normalize`: Loudness in LUFS to normalize input audio files to. Defaults to `-24` LUFS. Pass in `no` to turn off normalization.
- `-h`: Output the help message

**Examples**

Example running on a single file with default settings:

`uvic_music_extractor ./test_audio.wav ./output_features.csv`

Example running on a folder at a sampling rate of 48kHz and normalization to -18LUFs:

`uvic_music_extractor ./test_audio_folder/ ./output_features.csv --rate 48000 --normalize -18`


## Feature list

### Spectral

**Spectral Features**

A set of spectral features. Defaults to using a frame size of 2048 with half-overlap and a Hann window. Frame-by-frame results are summarized using the mean and standard deviation.
- spectral centroid, spectral spread, spectral skewness, spectral kurtosis, spectral flatness, spectral entropy, rolloff 85%, rolloff 95%, harsh energy, low frequency energy, dissonance, inharmonicity. See [Essentia documenation](https://essentia.upf.edu/algorithms_reference.html) for more information.

**Spectral Flux**

Spectral Flux Features. Performs spectral flux analysis using sub-bands from an octave spaced filter bank decomposition. Defaults to a 10-band octave filterbank with the lowest band from 0-50Hz. Uses a 2048 window with half overlap. [6,7]

### Loudness & Dynamics

**Crest Factor**

Peak-to-average ratio where peak is the the maximum amplitude level and average is the RMS value. Computed over the entire input signal as well as with frame-by-frame processing using frame sizes of 100ms and 1s. Frame-by-frame results are summarized using the mean and standard deviation over frames. [8]

**Loudness Range**

Loudness range is computed from short-term loudness values. It is defined as the difference between the estimates of the 10th and 95th percentiles of the distribution of the loudness values with applied gating [9]. See Essentia documentation for more information: https://essentia.upf.edu/reference/std_LoudnessEBUR128.html

**Microdynamics (LDR)**

LDR is a measurement of microdynamics. It is computed by taking the difference between loudness measurements using a fast integration time and a slow integration time, then computing the maximum and 95 percentile value from those results [10].

**Peak-to-loudness**

Peak-to-loudness is computed by taking the ratio between the true peak amplitude and the overall loudness [10].

**Top1dB**

Ratio of audio samples in the range [-1dB, 0dB] [11].

**Dynamic Spread**

Measure of the loudness spread across the audio file. The difference between the loudness (using Vickers algorithm) for each frame compared to the average loudness of the entire track is computed. Then, the average of that is computed [12].

### Distortion

Set of distortion features -- computes a probability mass function (pmf) on audio samples using a histogram with 1001 bins. Several statistics are computed on the resulting pmf including the centroid, spread, skewness, kurtosis, flatness, and the 'gauss' feature. 'Gauss' is a measurement of the gaussian fit of the the pmf [1, 13].

### Stereo Features

Side-to-mid ratio and left/right imbalance [2]

**Phase Correlation**
Calculates the correlation coefficient between the left and right channel. Computes phase correlation over the entire track as well as using short-time processing sing a frame size of 2048 with no overlap. Short-time results are summarized using mean and standard deviation. A value of +1 indicates the left and right channels are completely correlated (the same), a value of 0 indicates the left and right channels are not correlated, and a value of -1 indicates the left and right channels are completely out of phase.

**Stereo Spectrum**
Stereo Spectrum Features. Panning features computed using spectrums from the left and right audio channels. Returns features from the entire spectrum as well as three sub-bands which include 0-250Hz, 250-2800Hz, and 2800+ Hz [14].

### Other

Zero-crossing rate [15]

## References

[1] Wilson, A. D., and B. M. Fazenda. "Perception & evaluation of audio quality in music production." Proc. of the 16th Int. Conference on Digital Audio Effects (DAFx-13). 2013.

[2] Man, B. D., et al. "An analysis and evaluation of audio features for multitrack music mixtures." (2014).

[3] De Man, Brecht, et al. "Perceptual evaluation of music mixing practices." Audio Engineering Society Convention 138. Audio Engineering Society, 2015.

[4] Wilson, Alex, and Bruno Fazenda. "Variation in multitrack mixes: analysis of low-level audio signal features." Journal of the Audio Engineering Society 64.7/8 (2016): 466-473.

[5] Exploring Preference for Multitrack Mixes Using Statistical Analysis of MIR and Textual Features

[6] Alluri, Vinoo, and Petri Toiviainen. "Exploring perceptual and acoustical correlates of polyphonic timbre." Music Perception 27.3 (2010): 223-242.

[7] Tzanetakis, George, and Perry Cook. "Multifeature audio segmentation for browsingand annotation." Proceedings of the 1999 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics. WASPAA'99 (Cat. No. 99TH8452). IEEE, 1999.

[8] https://en.wikipedia.org/wiki/Crest_factor

[9] EBU Tech Doc 3342-2011. "Loudness Range: A measure to supplement loudness normalisation in accordance with EBU R 128"

[10] Skovenborg, Esben. "Measures of microdynamics." Audio Engineering Society Convention 137. Audio Engineering Society, 2014.

[11] Tardieu, Damien, et al. "Production effect: audio features for recording techniques description and decade prediction." 2011.

[12] Vickers, Earl. "Automatic long-term loudness and dynamics matching." Audio Engineering Society Convention 111. Audio Engineering Society, 2001.

[13] Wilson, Alex, and Bruno Fazenda. "Characterisation of distortion profiles in relation to audio quality." Proc. of the 17th Int. Conference on Digital Audio Effects (DAFx-14). 2014.

[14] Tzanetakis, George, Randy Jones, and Kirk McNally. "Stereo Panning Features for Classifying Recording Production Style." ISMIR. 2007.

[15] https://essentia.upf.edu/reference/std_ZeroCrossingRate.html

## Development
**Custom Scripts and Extractors**

This package is setup to be extensible and to support usage of the individual audio feature extractors independently of the main script. All extractor classes are available at `src/uvic_music_extractor/extractors.py`. See the main script: `scripts/uvic_music_extractor` for an example on how to put together the extractors into a script. Custom extractors can also be built by inheriting from the `ExtractorBase` class in the `extractors` module. See the implemented extractors in the `extractors` module for examples on how this works.
