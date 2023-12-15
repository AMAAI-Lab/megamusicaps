# Mega Music Captions

[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)](https://github.com/AMAAI-Lab/megamusicaps)

This repository contains scripts for generating captions for audio snippets using a combination of feature extraction and natural language processing. The system utilizes Essentia for audio feature extraction and OpenAI's GPT-3.5 Turbo for caption generation.

## Usage

To use the music captioning system, follow these steps:

#### 1. Set Up Conda Environment

Create and activate a conda environment using the provided musicaps_env.yml file. Ensure that you have changed the prefix in the bottom line of the yaml to the path to your conda environments before running the following commands

```
conda env create -f musicaps_env.yml
conda activate musicaps
```

#### 2. Configuration

Create / modify the configuration file (config/caption_generator_config.yaml) with the necessary parameters. Example configuration is provided in the configs section of this README.

#### 3. Run the Captioning System

Run the main script (main.py) with the path to your configuration file as a command-line argument:

```
    python3 main.py config.yaml
```

#### 4. Generated Captions

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
