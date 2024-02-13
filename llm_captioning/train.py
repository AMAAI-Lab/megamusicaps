import time
import argparse
import json
import logging
import math
import os

import datasets
import numpy as np
import pandas as pd
import wandb
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from accelerate import DistributedDataParallelKwargs

from tqdm.auto import tqdm

import transformers
from transformers import SchedulerType, get_scheduler
from processor import LanguageProcessor
from models.audio import AudioLLM
from dataloader import InstructDataset
from transformers import PretrainedConfig

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune multimodal language model TRIMERA.")
    parser.add_argument(
        "--train_file", type=str, default="data/train_audio.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default="data/valid_audio.json",
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--samples", type=int, default=-1,
        help="Samples to use for training and validation"
    )
    parser.add_argument(
        "--mode", type=int, default=2,
        help="How to construct LLM inputs"
    )
    parser.add_argument(
        "--audio_model", default="MIT/ast-finetuned-audioset-10-10-0.4593", type=str, 
        help="which audio model: ast or clap or encodec"
    )
    parser.add_argument(
        "--language_model", default="/mnt/data1/deep/trimera/vicuna-7b", type=str, 
        help="Language model name."
    )
    parser.add_argument(
        "--phase", default="pretrain", type=str, 
        help="Training phase: pretrain or finetune or finetune-align"
    )
    parser.add_argument(
        "--stage", default="2", type=str, 
        help="Debugging stage for run checking."
    )
    parser.add_argument(
        "--pretrained_path", default="", type=str, 
        help="Saved directory containing pytorch_model.bin from alignment stage."
    )
    parser.add_argument(
        "--precision", default="full", type=str, 
        help="Train on half or full precision."
    )
    parser.add_argument(
        "--lora", action="store_true", default=False,
        help="Use LoRA fine-tuning.",
    )
    parser.add_argument(
        "--max_lm_input_length", type=int, default=768,
        help="Max number of input tokens in the LLM"
    )
    parser.add_argument(
        "--max_lm_output_length", type=int, default=128,
        help="Max number of output tokens in the LLM"
    )
    parser.add_argument(
        "--num_audio_tokens", type=int, default=32,
        help="Number of tokens from the audio projection layer"
    )
    parser.add_argument(
        "--projection_depth", type=int, default=2,
        help="Projection transformer layer depth"
    )
    parser.add_argument(
        "--projection_dim", type=int, default=768,
        help="Projection transformer layer dimension"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=2,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=10,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )
    parser.add_argument(
        "--save_every", type=int, default=1,
        help="Save model after every how many epochs when checkpointing_steps is set to best."
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir
    
    ddp_scaler = DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_scaler], **accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    dataset_name = args.train_file.split("/")[-2].upper()
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/audio/{}".format(str(int(time.time())))            
            os.makedirs("saved", exist_ok=True)
            os.makedirs("saved/audio", exist_ok=True)
            os.makedirs(args.output_dir, exist_ok=True)
            
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(project="Trimera")
        
    accelerator.wait_for_everyone()
        
    if "t5" in args.language_model or "flan-ul2" in args.language_model:
        decoder_only_language_model = False
    else:
        decoder_only_language_model = True
        
    audio_encoder_model = args.audio_model
        
    config = {
        "phase": args.phase,
        "audio_model": audio_encoder_model, "language_model": args.language_model, "lora": args.lora,
        "use_decoder_only_language_model": decoder_only_language_model,
        "audio_projection_dim": args.projection_dim, "t_depth": args.projection_depth,
        "num_query_tokens": args.num_audio_tokens,
        "precision": args.precision,
    }
    
    model_config = PretrainedConfig.from_dict(config)
    model = AudioLLM(model_config)
        
    processor = LanguageProcessor(args, model_config)
    accelerator.print ("Loaded audio model from: {}".format(audio_encoder_model))
    accelerator.print ("Loaded language model from: {}".format(args.language_model))
    if args.lora:
        accelerator.print ("Using LoRA fine-tuning on LLM.")
    
    if args.pretrained_path != "" :
        weights = torch.load(args.pretrained_path + "/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(weights, strict=False)
        accelerator.print("Successfully loaded pretrained weights from: {}".format(args.pretrained_path))
        
    # Initialize dataloaders
    with accelerator.main_process_first():
        train_dataset = InstructDataset(args.train_file, args.samples)
        eval_dataset = InstructDataset(args.validation_file, args.samples)
        accelerator.print("Num instances in train: {}, validation: {}".format(train_dataset.get_num_instances(), eval_dataset.get_num_instances()))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size)

    # Trainable Parameters
    for name, params in model.named_parameters():
        if f"audio_model" in name:
            if "audio" in args.phase:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif "language_model" in name:
            if "finetune" in args.phase:
                # Train only norm, bias and lora parameters for LLM
                # We can also do full fine-tuning of the LLM but that would be expensive
                if "norm" in name or "bias" in name or "lora" in name:
                    if args.precision == "half":
                        params.data = params.data.float()
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False
        else:
            # Always keep projection layer trainable
            params.requires_grad = True
    
    # Optimizer
    optimizer_parameters = list(params for params in model.parameters() if params.requires_grad)
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("trimera", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0
    best_loss = np.inf
    device = accelerator.device

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):
                file_paths, questions, answers = batch
                tokenized_batch = processor.tokenize(questions, answers, device)
                tokenized_batch[f"audio_paths"] = file_paths
                # with torch.cuda.amp.autocast():
                output = model(**tokenized_batch, return_dict=True)
                loss = output["loss"]
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        unwrapped_model = accelerator.unwrap_model(model).to("cpu")
                        accelerator.save_state(output_dir)
                        unwrapped_model = accelerator.unwrap_model(model).to(accelerator.device)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()

        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                file_paths, questions, answers = batch
                tokenized_batch = processor.tokenize(questions, answers, device)
                tokenized_batch[f"audio_paths"] = file_paths
                # with torch.cuda.amp.autocast():
                output = model(**tokenized_batch, return_dict=True)
                val_loss = output["loss"]
                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1,
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
            result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)

            wandb.log(result)

            result_string = "Epoch: {}, Loss Train: {}, Val: {}\n".format(epoch+1, result["train_loss"], result["val_loss"])
            
            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["val_loss"] < best_loss:
                best_loss = result["val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False

        if args.with_tracking:
            accelerator.log(result, step=completed_steps)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model).to("cpu")
        trainable_state_dict = {name: params for name, params in unwrapped_model.named_parameters() if params.requires_grad}
        
        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                save_dir = "{}/{}".format(args.output_dir, "best")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(trainable_state_dict, save_dir + "/pytorch_model.bin")
                
            if (epoch + 1) % args.save_every == 0:
                save_dir = "{}/{}".format(args.output_dir, "epoch_" + str(epoch+1))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(trainable_state_dict, save_dir + "/pytorch_model.bin")
                
            if (epoch + 1) == args.num_train_epochs:
                save_dir = "{}/{}".format(args.output_dir, "epoch_" + str(epoch+1))
                os.makedirs(save_dir, exist_ok=True)
                torch.save(trainable_state_dict, save_dir + "/pytorch_model.bin")

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            save_dir = "{}/{}".format(args.output_dir, "epoch_" + str(epoch+1))
            os.makedirs(save_dir, exist_ok=True)
            torch.save(trainable_state_dict, save_dir + "/pytorch_model.bin")
            
        unwrapped_model = accelerator.unwrap_model(model).to(accelerator.device)

if __name__ == "__main__":
    main()
