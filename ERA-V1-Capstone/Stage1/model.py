import os
import gc
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule
from torch.cuda.amp import autocast

device = 'cuda'

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
    
class CaptionGeneratorGPT(LightningModule):
    def __init__(self, clip_embed_size, llm_embed_size, num_validation_examples=2, num_training_steps=100000):
        super().__init__()
        self.llm_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                            trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        self.projection_layer = nn.Linear(clip_embed_size, llm_embed_size)
        self.resnet_layer = SimpleResBlock(llm_embed_size)

        self.num_validation_examples = num_validation_examples
        self.num_training_steps = num_training_steps
        self.optimizer = None
        self.scheduler = None
        self.scheduler_dict = {}
        self.this_step_train_loss = None
        self.predicted_list = []
        self.expected_list = []
        self.save_hyperparameters(ignore=['loss_criterion', 'epoch'])
        self.train_loss_values = []
        self.val_loss_values = []
        self.COMMENT_TOKEN_ID = 23893
        self.IMAGE_TOKEN_ID = 5159
        self.EOS_TOKEN_ID = 50256
        self.image_embedding_dim = 49

        for param in self.llm_model.parameters():
            param.requires_grad = False

    
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
            
    def forward(self, batch):
        image_embeds = batch["image_embeddings"]
        targets = batch["tokenized_captions"]
        x = self.projection_layer(image_embeds)
        x = self.resnet_layer(x)
        outputs_dict = self.llm_model(inputs_embeds = x,
                                     labels = targets,
                                     return_dict = True) 
        
        del x
        del image_embeds
        return outputs_dict
    
    def training_step(self, batch):
        inputs_embeds, labels = self.preprocess_inputs(batch)
        outputs_dict = self.llm_model(inputs_embeds = inputs_embeds,
                                      labels = labels,
                                      return_dict = True) 
        loss = outputs_dict.loss
        pred_loss = self.loss_fn(outputs_dict.logits.squeeze(0), labels.squeeze(0))
        del inputs_embeds
        gc.collect()
        torch.cuda.empty_cache()
        self.log("train_loss", pred_loss.item(), prog_bar=True)
        self.log("train_loss_auto", loss.item(), prog_bar=True)
        return pred_loss
    
    def generate(self, x):
        llm_inputs = self.get_llm_inputs(x)
        
        comment_token = torch.tensor(self.COMMENT_TOKEN_ID).to(device)
        comment = self.llm_model.model.embed_tokens(comment_token).unsqueeze(0).unsqueeze(0).to(device) # 
        im_start_token = torch.tensor(self.IMAGE_TOKEN_ID).to(device)
        im_start = self.llm_model.model.embed_tokens(im_start_token).unsqueeze(0).unsqueeze(0).to(device) # 
        # prepare input embeddings
        inputs_embeds = torch.cat([im_start, # <IM> [1 x 1 x 2560]
                                  llm_inputs, # [1 x 49 x 2560]
                                  comment, # [1 x 1 x 2560]
                                  ], dim=1) # total dim: (B, 64, 2560)  
        with torch.no_grad():
            pred_logits = self.llm_model.generate(inputs_embeds = inputs_embeds, max_new_tokens=64)
            generated_text = self.tokenizer.batch_decode(pred_logits, skip_special_tokens=True)[0]
        return generated_text
    
    def get_llm_inputs(self, image_embeds):
        resnet_inputs = self.projection_layer(image_embeds)
        return self.resnet_layer(resnet_inputs)
    
    def preprocess_inputs(self, batch):
        """
        input format: <IMAGE_TOKEN> [1 x 49 x 2560] <COMMENT_TOKEN> [1 x 13 x 2560]
        targets: [1 x 49 x 2560] <COMMENT_TOKEN> [1 x 14 x 2560]
        """
        caption_tokens = batch['tokenized_caption']  # (1, 13)
        image_embeddings_llm_input = self.get_llm_inputs(batch['image_embeddings'])# (1, 1, 49, 2560)
        device = image_embeddings_llm_input.device
        im_start_token = torch.tensor(self.IMAGE_TOKEN_ID).to(device)
        comment_token = torch.tensor(self.COMMENT_TOKEN_ID).to(device)

        im_start = self.llm_model.model.embed_tokens(im_start_token).unsqueeze(0).unsqueeze(0).to(device) # 
        comment = self.llm_model.model.embed_tokens(comment_token).unsqueeze(0).unsqueeze(0).to(device) # 
        caption_embeddings = self.llm_model.model.embed_tokens(caption_tokens).to(device)
        # prepare input embeddings
        inputs_embeds = torch.cat([im_start, # <IM> [1 x 1 x 2560]
                                  image_embeddings_llm_input, # [1 x 49 x 2560]
                                  comment, # [1 x 1 x 2560]
                                  caption_embeddings, # [1 x 13 x 2560]
                                  ], dim=1) # total dim: (B, 64, 2560)  
        # prepare labels
        labels = torch.cat([torch.tensor([self.IMAGE_TOKEN_ID]*self.image_embedding_dim).unsqueeze(0).to(device), # [1 x 49] 
                            torch.tensor([self.COMMENT_TOKEN_ID]).unsqueeze(0).to(device), # [1 x 1]
                            caption_tokens.to(device), # [1 x 13]
                            torch.tensor([self.EOS_TOKEN_ID]).unsqueeze(0).to(device), # [1 x 1]
                            ], dim=1) # total dim: (1, 64)  
        return inputs_embeds, labels.to(device)

    
    def loss_fn(self, preds, targets):
        loss = 0
        seq_len = targets.shape[0]
        for cnt in range(seq_len):
            this_loss = torch.nn.functional.cross_entropy(preds[cnt,:], targets[cnt], label_smoothing=0.1)
            loss+=this_loss
        return loss/seq_len
    

    def evaluate(self,batch, stage):
        if stage:
            predicted = self.generate(batch["image_embeddings"])
            print("*****************************************")
            input_text = f"{f'caption: ' :>12}{batch['caption'][0]}"
            pred_text = f"{f'predicted: ' :>12}{predicted}"
            print(f"*****************************************\n{input_text}\n{pred_text}")

        return predicted
    
    def validation_step(self, batch, batch_idx):
        if batch_idx < self.num_validation_examples:
            predicted = self.evaluate(batch, "val")

    def test_step(self, batch):
        self.evaluate(batch, "test")
                    
        