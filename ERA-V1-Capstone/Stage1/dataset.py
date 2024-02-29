import os
import gc
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json
from pathlib import Path
import requests


class CocoImagesAndCaptionsDataset(Dataset):

    def __init__(self,
                 image_ids,
                 raw_images_list,
                 captions,
                 clip_model,
                 clip_processor,
                 tokenizer,
                 seq_len=64):
        super().__init__()
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.tokenizer = tokenizer
        self.eos_token = 50256
        self.image_token = 5159
        self.seq_len = seq_len
        self.image_ids = image_ids
        self.raw_images_list = raw_images_list
        self.captions = captions
        self.image_embedding_size = 49 
        self.max_caption_len = self.seq_len - (1+self.image_embedding_size+2)
        

    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):

        # get image embeddings
        image = Image.open(requests.get(self.raw_images_list[int(self.image_ids[idx])], stream=True).raw)
        processed_image = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.clip_model(**processed_image)
        img_embds =  outputs.last_hidden_state.squeeze()[1:,:]
        
        # get tokenized caption
        image_id = "{:012d}".format(int(self.image_ids[idx]))
        caption = self.captions[int(image_id)]
        tokenized_caption = self.tokenize_caption(caption)
        
        return {
            "image_embeddings": img_embds,
            "image_url": self.raw_images_list[int(self.image_ids[idx])],
            "caption": caption,
            "tokenized_caption": tokenized_caption
        }
    
    def tokenize_caption(self, caption):
        tokenizer_output = self.tokenizer(caption, return_tensors="pt", return_attention_mask=False)
        tokenized_caption = tokenizer_output['input_ids'].squeeze()
        if len(tokenized_caption) > self.max_caption_len:
            tokenized_caption = torch.cat([tokenized_caption[:self.max_caption_len], torch.tensor([self.eos_token], dtype=torch.int64)], dim=0)
        else: 
            num_padding_tokens = self.max_caption_len - len(tokenized_caption) + 1
            tokenized_caption = torch.cat([tokenized_caption, torch.tensor([self.eos_token] * num_padding_tokens, dtype=torch.int64)], dim=0)
        return tokenized_caption
    
    def collate_fn(self, batch):
        max_len = max(x["token_len"] for x in batch)

        captions_list = []
        image_embeddings_list = []
        tokenized_captions_list = []
        image_url_list = []

        for cnt, x in enumerate(batch):
            num_padding_tokens = max(0, max_len - len(x["tokenized_caption"]))+1 
            image_tokens = torch.tensor([self.image_token]*self.image_embedding_size,dtype=torch.int64)

            batch_x = torch.cat(
                [
                    image_tokens,
                    x['tokenized_caption'],
                    torch.tensor([self.eos_token] * num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            )
            tokenized_captions_list.append(batch_x)
            image_embeddings_list.append(x['image_embeddings'])
            captions_list.append(x['caption'])
            image_url_list.append(x['image_url'])

        return {
            "image_embeddings": torch.vstack(image_embeddings_list),
            "tokenized_captions": torch.vstack(tokenized_captions_list),
            "captions": captions_list,
            "image_url_list": image_url_list,
        }