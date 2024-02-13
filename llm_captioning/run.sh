# The commands below are using the 2 GPU devices with indices 0 and 1. You can train on more GPUs with your specified ids. If you change the GPU indices then you also need to change 'gpu_ids' and 'num_processes' in the config file: configs/gpu_config.yaml, as this file is being used in the commands. 'num_processes' should be the number of gpus that you are using.

# Modify the --train_file and --validation_file for your use case. Each line should be a json with the followiing keys: {"audio", "question", "answer"}. 'audio' signifies the audio file path. 'question' signifies the instruction of the task that the llm should perform. 'answer' is your intended output fromm llm.
# For long music captions it could look like the following:
# {"audio": "path/to/your/music/file.wav", "question": "Generate a long caption for the music clip.", "answer": "A low sounding male voice is rapping over a fast paced drums playing a reggaeton beat along with a bass. Something like a guitar is playing the melody along. This recording is of poor audio-quality. In the background a laughter can be noticed. This song may be playing in a bar."}

# You can also add the music specific features as part of the question:
# {"audio": "path/to/your/music/file.wav", "question": "Generate a long caption for the music clip. It has the following features: gender: male; instrument: drums, guitar; tempo: high; .....", "answer": "A low sounding male voice is rapping over a fast paced drums playing a reggaeton beat along with a bass. Something like a guitar is playing the melody along. This recording is of poor audio-quality. In the background a laughter can be noticed. This song may be playing in a bar."}



# First align the audio encoder output and llm input space using --phase "pretrain"
# Only the projection parameters that glues together the audio encoder and llm would be trained in this phase
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/gpu_config.yaml \
train.py --phase "pretrain" --precision "half" \
--learning_rate 2e-4 --weight_decay 0.01 --num_train_epochs 5 --save_every 1 --num_audio_tokens 32 \
--train_file "data/train_audio.json" --validation_file "data/valid_audio.json" \
--per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_warmup_steps 500

# After alignment use --pretrained_path "path/to/your/aligned/model" to continue further fine-tuning with --phase "finetune"
# The projection parameters and some parameters of the llm would be trained in this phase
# Control what parameters to train in the block from line 278-299 in train.py
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/gpu_config.yaml \
train.py --phase "finetune" --pretrained_path "saved/audio/1707789642/epoch_5" --precision "full" \
--learning_rate 2e-4 --weight_decay 0.01 --num_train_epochs 5 --save_every 1 \
--train_file "data/train_audio.json" --validation_file "data/valid_audio.json" \
--per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_warmup_steps 500