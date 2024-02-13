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