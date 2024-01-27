# Mega Music Captions

[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)](https://github.com/AMAAI-Lab/megamusicaps)

This repository contains scripts for generating captions for audio snippets using a combination of feature extraction and natural language processing. The system utilizes Essentia for audio feature extraction and OpenAI's GPT-3.5 Turbo for caption generation.

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
wget google drive
```

Unzip pre-trained weights 
```
unzip spleeter_pretrained_weights.zip
```

### 2. Set Up Docker Container

Clone the docker image prepared with the required conda environment.
```
docker pull anuuu97c/megamusicaps:ubuntu_20.04_dev
```

Create a new docker container from the image. Ensure you replace '/path/to/dir/w/repository' with your parent folder of the repository

```
docker run -it --name megamusicaps_container -v /path/to/dir/w/repository:/proj megamusicaps:ubuntu_20.04_dev /bin/bash
```

### 3. Configuration

Create / modify the configuration file (config/caption_generator_config.yaml) with the necessary parameters. Example configuration is provided in the configs section of this README.

### 4. Run the Captioning System

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
