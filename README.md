# Mega Music Captions

[![Status](https://img.shields.io/badge/status-stable-green.svg)](https://github.com/AMAAI-Lab/megamusicaps) [![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)](https://github.com/AMAAI-Lab/megamusicaps)

## Description

Mega Music Captions is a tool for analyzing various musical features from audio clips and generating descriptions based on that analysis. It extracts details like key, chord progression, tempo, and genre, and can tell the difference between vocal and instrumental sections. It also creates simple captions describing the music using a natural language model.

The system is modular, so you can easily turn different features on or off and customize the captions with different APIs. Whether you're using it for research, creative projects, or just to better understand music, Mega Music Captions offers a practical way to analyze and describe audio.

## Features and Models

Mega Music Captions uses advanced models for different music analysis tasks. Here's a quick rundown of the main features:

1. **Key Detection**: 
   - **Model**: CNNs with Directional Filters ([Schreiber and Müller, 2019](https://github.com/hendriks73/key-cnn))
   - **Description**: Identifies the key of the music using a CNN-based model.
   
2. **Chord Detection**:
   - **Model**: Bidirectional Transformer ([Park et al., 2019](https://github.com/jayg996/BTC-ISMIR19))
   - **Description**: Extracts chord sequences from the audio using deep auditory models and Conditional Random Fields.

3. **Tempo Estimation & Downbeat Transcription**:
   - **Model**: BeatNet: CRNN and Particle Filtering ([Heydari, Cwitkowitz, and Duan, 2021](https://github.com/mjhydri/BeatNet))
   - **Description**: Estimates the tempo and identifies downbeats in the music.

4. **Vocals / Instrumental Detection**:
   - **Model**: EfficientNet trained on Discogs ([Essentia Library](https://essentia.upf.edu))
   - **Description**: Classifies each clip as a track with vocals or an instrumental track

5. **Instrument, Mood, and Genre Detection**:
   - **Model**: Essentia's Jamendo Baseline Models ([Essentia Library](https://essentia.upf.edu))
   - **Description**: Classifies instruments, mood/themes, and genres using CNN-based models.

## Usage

To use the music captioning system, follow these steps:

### 1. Clone repository

Clone repository
```
git clone https://github.com/AMAAI-Lab/megamusicaps.git
```

Change directory into repository

```
cd megamusicaps
```

Download zip file with pre-trained weights required for source separation during pre-processing
```
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1yFs9ncHiV5-bI3_xrsgk7eL2QN9Hlldd' -O spleeter_pretrained_weights.zip
```

Unzip pre-trained weights 
```
unzip spleeter_pretrained_weights.zip
```

### 2. Set up environment

Follow the steps mentioned in [SETUP.md](SETUP.md) to prepare a conda environment to be used for running the Mega Music Captions tool

### 3. Configuration

Create / modify the configuration file (config/caption_generator_config.yaml) with the necessary parameters. Example configuration file and details on configuration options are provided in the configs section of this README.

### 4. Run the Captioning System

#### Environment

Activate conda environment in a new terminal
```
conda activate megamusicaps
```

#### Preprocess

Generate raw input json file for all mp3 files in given directory. Set directories with mp3 files in utils/create_input_json.py at line 23 to 27.

```
if __name__ == "__main__":
    # Example usage:
    input_directories = ["/proj/megamusicaps/samples/fma_top_downloads/", "/proj/megamusicaps/samples/generic_pop_audio/", "/proj/megamusicaps/samples/mtg_jamendo/"]
    output_json_file = "/proj/megamusicaps/files/audio_files.json"

```

Run create json with command

```
python utils/create_input_json.py
```

Run pre-processing to split audio into 30 second segments and split sources into 4 stems

```
python preprocess.py config/caption_generator_config.yaml
```


#### Caption audio

Run the main script (main.py) with the path to your configuration file as a command-line argument:

```
python main.py config/caption_generator_config.yaml
```

### 5. Generated Captions

The generated captions will be saved in the specified output file as JSON format.

## Configuration (config/caption_generator_config.yaml)

The configuration file includes settings for input/output paths, feature extractors, and the OpenAI GPT-3.5 Turbo model. Here is a breakdown of the configuration parameters:

    files:
	    input:  "/path/to/audio_files.json"
	    output:  "/path/to/captions.json"
	caption_generator:
		api_key:  "YOUR_OPENAI_API_KEY"
		model_id:  "gpt-3.5-turbo"
	extractors:
		mood_extractor:
			active:  True
			model:  "/path/to/mood_model.pb"
			model_metadata:  "/path/to/mood_model_metadata.json"
			embedding_model:  "/path/to/embedding_model.pb"
	# Add configurations for other extractors (genre, instrument, voice, gender, auto)...


### 6. Visualiser

The generated captions can be easily visualised using the following command

```
python display_tags.py files/captions.json (directory_with_megamusicaps_repo)/megamusicaps/
```

the requirements to run this are the following packages
- argparse
- tkinter
- pygame
- json

This will open a simple gui that can display the extracted feature values as stored in the captions.json, while playing the music.

There are buttons available to flipping to next or previous audio. Additionally there is a button to restart the audio if required.