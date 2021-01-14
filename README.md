# uvic-music-extractor
An audio feature extraction toolkit that was designed to provide a comprehensive selection of audio features for analyzing music. The choosen features were selected based on previous work conducted in the field of intelligent music production and audio recording research [1-5]. While most these features are currently available in existing audio feature extraction libraries, they are spread out over several different libraries written in different programming languages. The goal of this tool is to provide access to a large selection of commonly used features in a single Python package with limited dependencies. Specifcally, this package is written using [Essentia](https://essentia.upf.edu/index.html).

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

### References

[1] Wilson, A. D., and B. M. Fazenda. "Perception & evaluation of audio quality in music production." Proc. of the 16th Int. Conference on Digital Audio Effects (DAFx-13). 2013.

[2] Man, B. D., et al. "An analysis and evaluation of audio features for multitrack music mixtures." (2014).

[3] De Man, Brecht, et al. "Perceptual evaluation of music mixing practices." Audio Engineering Society Convention 138. Audio Engineering Society, 2015.

[4] Wilson, Alex, and Bruno Fazenda. "Variation in multitrack mixes: analysis of low-level audio signal features." Journal of the Audio Engineering Society 64.7/8 (2016): 466-473.

[5] Exploring Preference for Multitrack Mixes Using Statistical Analysis of MIR and Textual Features
