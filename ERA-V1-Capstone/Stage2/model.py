import os
import gc
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Union, List
from torch.cuda.amp import autocast
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import Callback
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
import torchinfo
import torchmetrics
import wandb

device = 'cuda'

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['projection_layer']
    for name, module in model.named_modules():
        #if any(mm_keyword in name for mm_keyword in multimodal_keywords):
        #    continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def model_summary(model, input_size):
    torchinfo.summary(model, 
                      input_size = input_size, 
                      batch_dim=0, 
                      col_names=("kernel_size",
                                 "input_size",
                                 "output_size",
                                 "num_params",
                                 "mult_adds"),
                       verbose=1,) 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)



class LitMultiModalGPT(LightningModule):
    """
    Pytorch Lightning module for Transformer

    """
    def __init__(self,
                 projection_layer_in_channels,
                 projection_layer_out_channels,
                 quantization_config,
                 num_validation_examples=3,
                 num_training_steps=100000):
        super(LitMultiModalGPT, self).__init__()
        self.quantization_config = quantization_config
        self.num_validation_examples = num_validation_examples
        #self.load_in_4bit = load_in_4bit
        self.num_training_steps = num_training_steps
        self.scheduler = None
        self.scheduler_dict = {}
        self.optimizer = None
        self.this_step_train_loss = None
        self.predicted_list = []
        self.expected_list = []
        self.save_hyperparameters(ignore=['loss_criterion', 'epoch'])
        #self.model = self.build_model()
        self.projection_layer = nn.Linear(projection_layer_in_channels, projection_layer_out_channels)
        self.resnet_layer = SimpleResBlock(projection_layer_out_channels)
        self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", quantization_config = quantization_config, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.train_loss_values = []
        self.val_loss_values = []
        self.COMMENT_TOKEN_ID = 23893
        self.IMAGE_TOKEN_ID = 5159
        self.EOS_TOKEN_ID = 50256
        self.image_embedding_dim = 49
        self.prepare_model_for_finetuning()
        self.print_all_trainable_params()


    def prepare_model_for_finetuning(self):
        # prepare for 4 bit training
        #self.llm_model.config.torch_dtype=torch.float32 
        self.llm_model = prepare_model_for_kbit_training(self.llm_model, use_gradient_checkpointing=False)
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                'k_proj',
                'v_proj',
                'fc1',
                'fc2'
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        #self.llm_model.to(torch.float16)
        self.llm_model = get_peft_model(self.llm_model, lora_config)

        # convert the projection layer to float 16
        #self.projection_layer.to(dtype=torch.float16)
        #self.resnet_layer.to(dtype=torch.float16)

        # convert all modules in LLM to float 16
        #for name, module in self.llm_model.named_modules():
        #   if 'norm' in name:
        #       module = module.to(torch.float32)
        #   if 'lm_head' in name or 'embed_tokens' in name:
        #       if hasattr(module, 'weight'):
        #           if module.weight.dtype == torch.float32:
        #               module = module.to(torch.float16)


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    
    def set_scheduler_dict(self, scheduler, freq='step'):
        self.scheduler = scheduler
        self.scheduler_dict = {
            "scheduler": self.scheduler,
            "interval": freq,
        }

    def configure_optimizers(self):
        if self.scheduler_dict:
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler_dict}
        return {"optimizer": self.optimizer}


    def print_all_trainable_params(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.llm_model.print_trainable_parameters()
        # print trainable paramaeters
        print("Number of Training Parameters")
        print("********************************")
        print(f"Projection Layer:{count_parameters(self.projection_layer)}")
        print(f"Phi Model:{count_parameters(self.llm_model)}")
        print("********************************")

    def get_llm_inputs(self, x):
        resnet_inputs = self.projection_layer(x)
        return self.resnet_layer(resnet_inputs)

    def generate(self, x):
        proj_outs = self.get_llm_inputs(x)
        #device = proj_outs.device
        comment_token = torch.tensor(self.COMMENT_TOKEN_ID).to(device)
        comment = self.llm_model.model.model.embed_tokens(comment_token).unsqueeze(0).unsqueeze(0).to(device) # 
        im_start_token = torch.tensor(self.IMAGE_TOKEN_ID).to(device)
        im_start = self.llm_model.model.model.embed_tokens(im_start_token).unsqueeze(0).unsqueeze(0).to(device) # 
        # prepare input embeddings
        inputs_embeds = torch.cat([im_start, # <IM> [1 x 1 x 2560]
                                  proj_outs, # [1 x 49 x 2560]
                                  comment, # [1 x 1 x 2560]
                                  ], dim=1) # total dim: (B, 64, 2560)  
        with torch.no_grad():
            pred_logits = self.llm_model.generate(inputs_embeds = inputs_embeds, max_new_tokens=64)
            generated_text = self.tokenizer.batch_decode(pred_logits, skip_special_tokens=True)[0]
        return generated_text
    
    
    def evaluate(self,batch, stage):
        if stage:
            predicted = self.generate(batch['image_embeddings'])
            #self.predicted_list.append(predicted)
            #self.expected_list.append(batch['caption'])
            # print the source, target, and the model output
            #print("*****************************************")
            input_text = f"{f'question: ' :>12}{batch['question'][0]}"
            pred_text = f"{f'predicted: ' :>12}{predicted}"
            print(f"*****************************************\n{input_text}\n{pred_text}")
            # log a W&B Table that has a text caption, an image and audio
            #columns = ["step", "caption", "prediction"]

            # data should be a list of lists
            #val_data_log.append([self.global_step,input_text , pred_text])
            ## log the Table
            #wandb_logger.log_table(key="val_samples", columns=columns, data=val_data_log)
        return predicted


    def preprocess_inputs(self, batch):
        """
        input format: <IMAGE_TOKEN> [1 x 49 x 2560] <COMMENT_TOKEN> [1 x 13 x 2560]
        targets: [1 x 49 x 2560] <COMMENT_TOKEN> [1 x 14 x 2560]
        """
        question_tokens = batch['ques_tokenized']  # (1, 13)
        answer_tokens = batch['ans_tokenized']  # (1, 13)
        im_start_token = torch.tensor(self.IMAGE_TOKEN_ID).to(device)
        comment_token = torch.tensor(self.COMMENT_TOKEN_ID).to(device)

        # project image embeddings to llm dim
        image_embeddings_llm_input = self.get_llm_inputs(batch['image_embeddings'])# (1, 1, 49, 2560)
        #device = image_embeddings_llm_input.device

        im_start = self.llm_model.model.model.embed_tokens(im_start_token).unsqueeze(0).unsqueeze(0).to(device) # 
        comment = self.llm_model.model.model.embed_tokens(comment_token).unsqueeze(0).unsqueeze(0).to(device) # 
        question_embeddings = self.llm_model.model.model.embed_tokens(question_tokens).to(device)
        # prepare input embeddings
        #print(im_start.shape, image_embeddings_llm_input.shape, comment.shape, question_embeddings.shape, answer_tokens.shape, question_tokens.shape)
        inputs_embeds = torch.cat([im_start, # <IM> [1 x 1 x 2560]
                                  image_embeddings_llm_input, # [1 x 49 x 2560]
                                  comment, # [1 x 1 x 2560]
                                  question_embeddings, # [1 x 13 x 2560]
                                  ], dim=1) # total dim: (B, 64, 2560)  
        # prepare labels
        labels = torch.cat([torch.tensor([self.IMAGE_TOKEN_ID]*self.image_embedding_dim).unsqueeze(0).to(device), # [1 x 49] 
                            torch.tensor([self.COMMENT_TOKEN_ID]).unsqueeze(0).to(device), # [1 x 1]
                            answer_tokens, # [1 x 1]
                            torch.tensor([self.EOS_TOKEN_ID]).unsqueeze(0).to(device), # [1 x 1]
                            ], dim=1) # total dim: (1, 64)  
        #print(inputs_embeds.shape, labels.shape)
        return inputs_embeds, labels.to(device)

    
    def loss_fn(self, preds, targets):
        loss = 0
        seq_len = targets.shape[0]
        for cnt in range(seq_len):
            this_loss = torch.nn.functional.cross_entropy(preds[cnt,:], targets[cnt], label_smoothing=0.1)
            loss+=this_loss
        return loss/seq_len



    def training_step(self, batch):
        inputs_embeds, labels = self.preprocess_inputs(batch)
        outputs_dict = self.llm_model(inputs_embeds = inputs_embeds,
                                      labels = labels,
                                      return_dict = True) 
        loss = outputs_dict.loss
        pred_loss = self.loss_fn(outputs_dict.logits.squeeze(0), labels.squeeze(0))
        #print(f"auto loss: {loss}\tmanual loss: {pred_loss}")
        del inputs_embeds
        gc.collect()
        torch.cuda.empty_cache()
        self.log("train_loss", pred_loss.item(), prog_bar=True)
        self.log("train_loss_auto", loss.item(), prog_bar=True)
        return pred_loss

    
    def validation_step(self, batch, batch_idx):
        if batch_idx < self.num_validation_examples:
            predicted = self.evaluate(batch, "val")


    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
    