# Environment Setup


The repository has been tested with the following environment set up. To recreate, please follow the below installations in order


### Base

* Ubuntu 22.04
* Python 3.8

### Additional installations (in order of installation)

```
pip install essentia
pip install -f https://essentia.upf.edu/python-wheels/ essentia-tensorflow
pip install librosa
pip install git+https://github.com/CPJKU/madmom
sudo apt-get install portaudio19-dev
pip install pyaudio
pip install git+https://github.com/mjhydri/BeatNet
pip install openai

pip install torch
pip install pandas
pip install pyrubberband
pip install pyyaml
pip install mir_eval
pip install pretty_midi
pip uninstall pysoundfile
pip uninstall soundfile
pip install soundfile

pip install tensorflow==2.7.0

conda install -c conda-forge ffmpeg libsndfile
pip install spleeter
pip install httpx==0.26.0

pip install pydub
pip install protobuf==3.19.0

pip install jams
pip uninstall librosa
pip install librosa==0.6.2
pip uninstall resampy numba
pip install numba==0.48 resampy
```

### ENVIRONMENT VARIABLES

```
export TF_FORCE_GPU_ALLOW_GROWTH=true
```
