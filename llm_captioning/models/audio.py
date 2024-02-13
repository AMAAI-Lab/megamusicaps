from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

import torchaudio
from transformers.utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoConfig, AutoProcessor
from transformers import ASTModel, T5EncoderModel, EncodecModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

@dataclass
class AudioLLMOutput(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    audio_outputs: Optional[torch.FloatTensor] = None
    audio_query: Optional[torch.FloatTensor] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["audio_outputs", "audio_query", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )
    
    
class LLMOutput(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )
    

class AudioLLM(PreTrainedModel):
    def __init__(
        self, config,
    ):
        super().__init__(config)
        
        self.config = config
        self.phase = config.phase
        
        # 1. Main models
        self.audio_model_name = config.audio_model
        if "ast" in self.audio_model_name:
            self.audio_processor = AutoProcessor.from_pretrained(config.audio_model)
            self.audio_model = ASTModel.from_pretrained(config.audio_model)
            
        elif "encodec" in self.audio_model_name:
            self.audio_processor = AutoProcessor.from_pretrained(config.audio_model)
            self.audio_model = EncodecModel.from_pretrained(config.audio_model)
        
        if config.use_decoder_only_language_model:
            if config.precision == "half":
                language_model = AutoModelForCausalLM.from_pretrained(config.language_model, torch_dtype=torch.bfloat16)
            else:
                language_model = AutoModelForCausalLM.from_pretrained(config.language_model)
            config.lm_hidden_dim = language_model.config.hidden_size
            lora_task_type = "CAUSAL_LM"
        else:
            language_model = AutoModelForSeq2SeqLM.from_pretrained(config.language_model)
            config.lm_hidden_dim = language_model.config.d_model
            lora_task_type = "SEQ_2_SEQ_LM"
            
        if config.lora:
            linear_modules = find_all_linear_names(language_model)
            self.lora_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=linear_modules,
                bias="none", task_type=lora_task_type,
            )
            language_model = get_peft_model(language_model, self.lora_config)
        
        self.language_model = language_model

        # 2. Projection related layers
        config.num_query_layers = 1
        
        audio_dim = self.audio_model.config.hidden_size
        self.projection_enc = nn.Linear(audio_dim, config.audio_projection_dim)
        self.projection_enc_norm = nn.LayerNorm(config.audio_projection_dim)

        # self.audio_query = nn.Embedding(config.num_query_tokens, config.audio_projection_dim)
        
        projection_config = AutoConfig.from_pretrained("configs/projection-config.json")
        projection_config.d_model = config.audio_projection_dim
        projection_config.num_layers, projection_config.num_decoder_layers = config.t_depth, config.t_depth
        self.projection_t5 = AutoModel.from_config(projection_config).encoder
        
        self.projection_dec = nn.Linear(config.audio_projection_dim, config.lm_hidden_dim)
        self.projection_dec_norm = nn.LayerNorm(config.lm_hidden_dim)
        
        
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared
            
    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:
            # warn users about unexpected behavior when using multi-GPU + InstructBLIP + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script"
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility
            
            
    def pad_wav(
        self, waveform, segment_length
    ):
        waveform_length = len(waveform)

        if segment_length is None or waveform_length == segment_length:
            return waveform
        elif waveform_length > segment_length:
            return waveform[:segment_length]
        else:
            pad_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
            waveform = torch.cat([waveform, pad_wav])
            return waveform
            
    
    def read_wav_file(
        self, filename, segment_length=160000, new_freq=16000, padding=True
    ):
        waveform, sr = torchaudio.load(filename)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=new_freq)[0]
        if padding:
            waveform = self.pad_wav(waveform, segment_length).numpy()
        else:
            waveform = waveform.numpy()
        return waveform

            
    def encode_audio(
        self, audio_paths,
    ):  
        self.audio_model.eval()
        
        if "ast" in self.audio_model_name:
            waveform = [self.read_wav_file(path) for path in audio_paths]
            inputs = self.audio_processor(waveform, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.audio_model.dtype).to(self.audio_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.audio_model(**inputs)
                audio_features = outputs.last_hidden_state
                audio_mask = torch.ones(audio_features.shape[:-1], dtype=torch.long, device=audio_features.device)
                
        elif "encodec" in self.audio_model_name:
            all_embeddings = []
            waveform = [self.read_wav_file(path, new_freq=self.audio_processor.sampling_rate, padding=False) for path in audio_paths]
            for k in range(len(waveform)):
                inputs = self.audio_processor(
                    raw_audio=waveform[k], sampling_rate=self.audio_processor.sampling_rate, return_tensors="pt"
                )
                inputs = {k: v.to(self.audio_model.dtype).to(self.audio_model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.audio_model.encode(inputs["input_values"], inputs["padding_mask"])
                embeddings = outputs["embeddings"][0].transpose(1, 2)
                all_embeddings.append(embeddings)

            hidden_size = embeddings.shape[2]
            batch_token_lens = [embeddings.shape[1] for embeddings in all_embeddings]
            max_len = max(batch_token_lens)

            padding = [
                torch.zeros(1, max_len - item, hidden_size, dtype=embeddings.dtype, device=embeddings.device) 
                for item in batch_token_lens
            ]
            audio_features = [torch.cat([all_embeddings[k], padding[k]], 1) for k in range(len(all_embeddings))]
            audio_features = torch.cat(audio_features, 0)

            audio_mask = torch.tensor([[1] * item + [0] * (max_len - item) for item in batch_token_lens], 
                                      dtype=torch.long, device=audio_features.device)
            
        return audio_features, audio_mask

    
    def forward_audio(
        self, audio_paths,
    ):
        audio_feats, audio_mask = self.encode_audio(audio_paths)
        audio_feats = self.projection_enc_norm(self.projection_enc(audio_feats.float()))
        
        # audio_query = self.audio_query.weight.unsqueeze(0).repeat(len(audio_input_ids), 1, 1)
        # audio_query_mask = torch.ones(audio_query.shape[:-1], dtype=torch.long, device=audio_feats.device)
        # audio_query = torch.cat([audio_query, audio_feats], dim=1)
        # audio_query_mask = torch.cat([audio_query_mask, audio_mask], dim=1)
        
        audio_query = self.projection_t5(
            inputs_embeds=audio_feats, attention_mask=audio_mask
        )["last_hidden_state"]

        audio_query = audio_query[:, :self.config.num_query_tokens, :]
        audio_query = self.projection_dec(audio_query)
        audio_query = self.projection_dec_norm(audio_query)

        return audio_query, audio_feats

    def forward(
        self,
        audio_paths: list,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AudioLLMOutput]:
        
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the audio input through the audio model
        query_inputs, audio_feats = self.forward_audio(audio_paths)
        query_attention_mask = torch.ones(
            query_inputs.size()[:-1], dtype=torch.long, device=query_inputs.device
        )
        
        # step 2: use the language model, conditioned on the query outputs and the prompt
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([query_inputs.to(inputs_embeds.dtype), inputs_embeds.to(query_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([query_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.language_model.config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return AudioLLMOutput(
            loss=loss,
            logits=logits,
            audio_outputs=audio_feats,
            audio_query=query_inputs,
            language_model_outputs=outputs,
        )
    
    @torch.no_grad()
    def generate(
        self,
        audio_paths: list,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
            
        # step 1: forward the audio input through the audio model
        query_inputs, audio_feats = self.forward_audio(audio_paths)
        query_attention_mask = torch.ones(
            query_inputs.size()[:-1], dtype=torch.long, device=query_inputs.device
        )
        
        # step 2: use the language model, conditioned on the query outputs and the prompt
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([query_inputs.to(inputs_embeds.dtype), inputs_embeds.to(query_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([query_attention_mask.to(attention_mask.device), attention_mask], dim=1)
        
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
    