import torch
from fastchat.conversation import get_conv_template
from transformers import PretrainedConfig, AutoTokenizer, InstructBlipProcessor
    
class LanguageProcessor:
    def __init__(self, args, config):
        
        self.config = config
        
        if "trimera/vicuna-7b" in self.config.language_model:
            self.lm_tokenizer = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b").tokenizer
        else:
            self.lm_tokenizer = AutoTokenizer.from_pretrained(config.language_model)
        self.lm_tokenizer.add_eos_token = True
        
        self.max_lm_input_length = args.max_lm_input_length
        self.max_lm_output_length = args.max_lm_output_length            
        
    def tokenize(self, questions, answers, device, for_training=True):
        
        if "trimera/vicuna-7b" in self.config.language_model:
            modified_questions = questions
            
        elif "vicuna" in self.config.language_model or "WizardLM" in self.config.language_model:
            modified_questions = []
            for question in questions:
                conv = get_conv_template("vicuna_v1.1")
                conv.append_message(conv.roles[0], question)
                modified_questions.append(conv.get_prompt()[:-1])
        
        elif "WizardCoder" in self.config.language_model:
            modified_questions = []
            for question in questions:
                modified_questions.append(f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n###")
        
        else:
            modified_questions = questions
        
        
        inputs, labels = [], []
            
        for question, answer in zip(modified_questions, answers):
            question_encoded = self.lm_tokenizer.encode(question)[:-1][:self.max_lm_input_length] # before <eos>
            
            # add the assistant/response suffix at end so that is doesn't get truncated
            if "vicuna" in self.config.language_model or "WizardLM" in self.config.language_model:
                question_encoded += self.lm_tokenizer.encode("ASSISTANT:")[1:-1] # excluding <bos> and <eos>
            
            elif "WizardCoder" in self.config.language_model:
                question_encoded += self.lm_tokenizer.encode("Response:")[1:-1] # excluding <bos> and <eos>
                
            answer_encoded = self.lm_tokenizer.encode(answer)[1:][:self.max_lm_output_length] # after <bos>
            masked_question_encoded = [-100 for _ in range(len(question_encoded))]
            
            if for_training:
                inputs.append(question_encoded + answer_encoded)
                labels.append(masked_question_encoded + answer_encoded)
            else:
                inputs.append(question_encoded)
        
        if for_training:
            max_len = max([len(label) for label in labels])
            
            # Left Padding for Causal LM
            mask = torch.tensor([[0] * (max_len - len(input_ids)) + [1] * len(input_ids) for input_ids in inputs], dtype=torch.int64)
            inputs = torch.tensor([[self.lm_tokenizer.pad_token_id] * (max_len - len(input_ids)) + input_ids for input_ids in inputs], dtype=torch.int64)
            labels = torch.tensor([[-100 for _ in range(max_len - len(label))] + label for label in labels], dtype=torch.int64)
            batch = {
                "input_ids": inputs,
                "attention_mask": mask,
                "labels": labels
            }
            
        else:
            # For generation
            max_len = max([len(input_ids) for input_ids in inputs])
            # Left Padding for Causal LM
            mask = torch.tensor([[0] * (max_len - len(input_ids)) + [1] * len(input_ids) for input_ids in inputs], dtype=torch.int64)
            inputs = torch.tensor([[self.lm_tokenizer.pad_token_id] * (max_len - len(input_ids)) + input_ids for input_ids in inputs], dtype=torch.int64)
            batch = {
                "input_ids": inputs,
                "attention_mask": mask
            }
            
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch
    
    def decode(self, generated_ids):
        return self.lm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    