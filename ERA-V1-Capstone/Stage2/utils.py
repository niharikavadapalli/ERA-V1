import io 
import requests
import whisperx
import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from pathlib import Path
from transformers import AutoProcessor, CLIPVisionModel
from peft import LoraConfig
from model_finetune import LitMultiModalPhiFineTune
from torch.cuda.amp import autocast


device = 'cuda' if torch.cuda.is_available() else 'cpu'
COMMENT_TOKEN_ID = 23893
IMAGE_TOKEN_ID = 5159
EOS_TOKEN_ID = 50256

def prepare_inputs(multimodal_phi_model, proj_output=None, question_embeddings=None, batch_size=1):
    # define comment and im start tokens
    comment_token = torch.tensor(COMMENT_TOKEN_ID).repeat(batch_size, 1).to(device)
    comment = multimodal_phi_model.model.embed_tokens(comment_token).to(device) #

    im_start_token = torch.tensor(IMAGE_TOKEN_ID).repeat(batch_size, 1).to(device)
    im_start = multimodal_phi_model.model.embed_tokens(im_start_token).to(device) #

    if proj_output is None and question_embeddings is None:
        raise Exception("you need to provide an image, text, or audio input")
    if question_embeddings is None:
        # prepare input embeddings
        print(im_start.shape)
        print(proj_output.shape)
        print(comment.shape)
        inputs_embeds = torch.cat([im_start, # <IM> [B x 1 x 2560]
                                   proj_output, # [B x 49 x 2560]
                                   comment, # [B x 1 x 2560]
                                   ], dim=1) # total dim: (B, 64, 2560)
    else:
        # prepare input embeddings
        inputs_embeds = torch.cat([im_start, # <IM> [B x 1 x 2560]
                                   proj_output, # [B x 49 x 2560]
                                   comment, # [B x 1 x 2560]
                                   question_embeddings,
                                   ], dim=1) # total dim: (B, 64, 2560)
    return inputs_embeds


def generate_phi_responses(multimodal_phi_model, projection_layer, resnet_layer, tokenizer, batch, batch_size=1):
    question_embeddings = None
    proj_output = None
    if 'ques_tokenized' in batch:
        question_tokens = batch['ques_tokenized']
        question_embeddings = multimodal_phi_model.model.embed_tokens(question_tokens).to(device)

    if 'image_embeddings' in batch:
        image_embeddings = batch['image_embeddings']
        proj_output = projection_layer(image_embeddings).to(device)
        resnet_output = resnet_layer(proj_output).to(device)

    inputs_embeds = prepare_inputs(multimodal_phi_model, resnet_output, question_embeddings)

    with torch.no_grad():
        with autocast(True):
            pred_logits = multimodal_phi_model.generate(inputs_embeds = inputs_embeds, max_new_tokens=20)
            generated_text = tokenizer.batch_decode(pred_logits, skip_special_tokens=True, clean_up_tokenization_spaces=True, verbose=False)[0]
    return generated_text


def get_image_embeddings(image, model, preprocessor, device=None):
    processed_image = preprocessor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**processed_image)
    return outputs.last_hidden_state.squeeze()[1:,:].unsqueeze(0)

def tokenize_sentence(sentence, tokenizer):
    tokenizer_output = tokenizer(sentence, return_tensors="pt", return_attention_mask=False)
    tokenized_sentence = tokenizer_output['input_ids']
    return tokenized_sentence

def generate_embeddings_from_inputs(image, text, clip_model, clip_preprocessor, tokenizer):
    image_embeddings = get_image_embeddings(image, clip_model, clip_preprocessor)
    tokenized_sentence = tokenize_sentence(text, tokenizer)
    return {'image_embeddings': image_embeddings, 'ques_tokenized': tokenized_sentence}


def transcribe_audio(audio_model, audio_path):
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
    audio = whisperx.load_audio(audio_path)

    # 1. Transcribe with original whisper (batched)
    #audio = whisperx.load_audio(audio_file)
    print(audio)
    audio_result = audio_model.transcribe(audio)
    audio_text = ''
    for seg in audio_result['segments']:
        audio_text += seg['text']
    audio_text = audio_text.strip()
    return audio_text 