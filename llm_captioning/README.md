# Commands for Training
Use `bash run.sh`

The commands in `run.sh` are using the 2 GPU devices with indices 0 and 1. You can train on more GPUs with your specified ids. If you change the GPU indices then you also need to change `gpu_ids` and `num_processes` in the config file: `configs/gpu_config.yaml`, as this file is being used in the training commands. `num_processes` should be the number of gpus that you are using.

# Data
Modify the `--train_file` and `--validation_file` for your use case. Each line should be a json with the followiing keys: `{"audio", "question", "answer"}`. 

`audio` signifies the audio file path. `question` signifies the instruction of the task that the llm should perform. `answer` is your intended output from llm.

For long music captions it could look like the following:
```json
{
    "audio": "path/to/your/music/file.wav",
    "question": "Generate a long caption for the music clip.",
    "answer": "A low sounding male voice is rapping over a fast paced drums playing a reggaeton beat along with a bass. Something like a guitar is playing the melody along. This recording is of poor audio-quality. In the background a laughter can be noticed. This song may be playing in a bar."
}
```

You can also add the music specific features as part of the question:

```json
{
    "audio": "path/to/your/music/file.wav",
    "question": "Generate a long caption for the music clip. It has the following features: gender: male; instrument: drums, guitar; tempo: high; .....",
    "answer": "A low sounding male voice is rapping over a fast paced drums playing a reggaeton beat along with a bass. Something like a guitar is playing the melody along. This recording is of poor audio-quality. In the background a laughter can be noticed. This song may be playing in a bar."
}
```

# Language Models Used
The code was tested with `Salesforce/instructblip-vicuna-7b` and `WizardLM` models. New models can be added with some small changes in `processor.py` for input format handling and `models/audio.py` for model specific architecture handling.