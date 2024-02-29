# Multimodal GPT using Microsoft Phi model

For this project, we start by pretraining the Microsoft Phi model from the scratch and then finetune it to function as a multimodal language assistant. Once trained, the users can input an image, along with a textual question or an audio question, to ask details about the image and the model should be able to provide answers explaining it.

For purpose of this project, I used an Amazon EC2 instance with a GPU of 24 GB for ~30 hours that includes all three phases of training.

# Stage0 Pretrain:
In the pretraining step, I pretrained a [Pythia](https://github.com/EleutherAI/pythia) model using the [Red Pajama](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/pretrain_redpajama.md) dataset. 
We use the [LitGPT](https://github.com/Lightning-AI/lit-gpt/blob/main/README.md) repo to perform this. 

We only train it till the loss has reduced and demonstrate that given more GPU resources, we can continue to fully pretrain this model. Below are the training logs for this pretraining:

```
iter 0 step 1: loss 10.9833, LR: 0.000000, iter time: 1964.85ms (optimizer.step)
iter 100 step 101: loss 7.0670, LR: 0.000300, iter time: 625.12ms (optimizer.step)
iter 200 step 201: loss 6.1442, LR: 0.000600, iter time: 672.23ms (optimizer.step)
iter 300 step 301: loss 6.3695, LR: 0.000900, iter time: 663.85ms (optimizer.step)
iter 400 step 401: loss 6.4042, LR: 0.001200, iter time: 659.47ms (optimizer.step)
iter 500 step 501: loss 5.9287, LR: 0.001500, iter time: 656.67ms (optimizer.step)
iter 600 step 601: loss 6.1363, LR: 0.001800, iter time: 655.95ms (optimizer.step)
```
 
 # Stage 1: Train the LLM to know/understand the images
 ## Training a Projection Layer:
Once the pretraining is done, our goal is to train this LLM on an image dataset so that it understands the details in any given image. For this purpose, we used a pretrained [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) model equipped before the [Microsoft Phi 2](https://huggingface.co/microsoft/phi-2) model. When an image is sent to the model, it initially goes through this pretrained CLIP model that generates image embedding for this given image. This image embeddings are then fed to a Res-Net architecture based projection layer and then the outputs of this finetuned layer is sent to the pretrained Phi model. For this step, we use the [COCO 2017 dataset](https://cocodataset.org/#home) which has images and their captions with it. During this pretraining step, we make sure that the pretrained Phi model remains frozen and not trained, but we only train the projection layer such that the model predicts correct provided caption when we send corresponding image to it.


The pretraining was done for a total of ~100000 steps on an AWS EC2 instance with 24GB GPU memory. The plots below show the training loss and the step number. 

The pretraining was done for a total of ~60000 steps on an AWS EC2 instance with 24GB GPU memory. The plots below show the training loss and the step number. 
<img width="1437" alt="Screenshot 2024-02-28 at 6 35 04 PM" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/7cee2eb4-4a63-4b28-a019-66131cb07a10">

Below is a snapshot of some validation examples from the training run.

```
*****************************************
  *****************************************
   caption: a large boat on the beach near a body of water 
 predicted: A man is flying a kite on a beach.
Epoch 0:  83%|█████████████████████████████████████▎       | 98000/118169 [8:05:27<1:39:54,  3.36it/s, v_num=m0xd, train_loss=1.810, train_loss_auto=1.590]Step: 98001: train_loss=2.0594                                                                                                                             
Epoch 0:  83%|█████████████████████████████████████▌       | 98500/118169 [8:07:47<1:37:24,  3.37it/s, v_num=m0xd, train_loss=1.770, train_loss_auto=1.630]Step: 98501: train_loss=1.8592
Epoch 0:  84%|█████████████████████████████████████▋       | 99000/118169 [8:10:03<1:34:53,  3.37it/s, v_num=m0xd, train_loss=1.930, train_loss_auto=1.560The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
*****************************************
*****************************************
   caption: A skateboarder doing a jumping trick at a skate park.
 predicted: A young man is riding a skateboard on a concrete ledge
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
*****************************************
*****************************************
   caption: A shirtless man wearing a helmet and knee pads skateboarding on a halfpipe.
 predicted: A man is jumping on a skateboard.
Epoch 0:  84%|█████████████████████████████████████▋       | 99000/118169 [8:10:27<1:34:57,  3.36it/s, v_num=m0xd, train_loss=1.930, train_loss_auto=1.560]Step: 99001: train_loss=1.7936                                                                                                                             
Epoch 0:  84%|█████████████████████████████████████▉       | 99500/118169 [8:12:48<1:32:27,  3.37it/s, v_num=m0xd, train_loss=1.980, train_loss_auto=1.820]Step: 99501: train_loss=1.8967
Epoch 0:  85%|█████████████████████████████████████▏      | 100000/118169 [8:15:06<1:29:57,  3.37it/s, v_num=m0xd, train_loss=2.040, train_loss_auto=1.970The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
*****************************************
*****************************************
   caption: A man poses with a young boy who's holding a kite.
 predicted: A man and a boy standing on a bench in a park.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
*****************************************
*****************************************
   caption: A woman checks her phone while sitting on a wooden bench.
 predicted: A woman sitting on a bench reading a book.
Epoch 0:  85%|█████████████████████████████████████▏      | 100000/118169 [8:15:30<1:30:01,  3.36it/s, v_num=m0xd, train_loss=2.040, train_loss_auto=1.970]`Trainer.fit` stopped: `max_steps=100000` reached.                                                                                                         
Epoch 0:  85%|█████████████████████████████████████▏      | 100000/118169 [8:15:30<1:30:01,  3.36it/s, v_num=m0xd, train_loss=2.040, train_loss_auto=1.970]
```

## Finetuning the LLM with QLORA
After training the projection layer, we need to finetune the LLM to be able to answer questions from the images. For this, we use the [Llava Instruct 150k] (https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) dataset, which is based on COCO 2017 dataset but also contains pairs of questions and answers. 

To finetine the Phi model, we follow the [QLORA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) finetuning strategy. In this, we finetune the projection layer, along with the attention blocks in the Phi model. The finetuning process is explained here:

We send an image and question-answer pair as the training inputs to the finetune model. The image initially goes to CLIP model where we generate image embeddings similar to phase1 and are sent to the projection layer which gives out the image embeddings. We tokenize and compute text embedding for the current question and concatenate with the image embeddings we got in previous step which acts as input to our LLM. This new embeddings when sent to LLM generates output which needs to match the original answer we had for this image. During the training, I used cross entropy to calculate the loss between the LLM predicted answer and the original answer. Below is a chart that shows the loss during the finetuning.

<img width="1437" alt="Screenshot 2024-02-28 at 6 33 36 PM" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/3f429bb6-881c-4b32-8b39-b226e11003bc">

Even though I trained for about 20 hours for this task, it looks like the model was not able to fully learn and might need additional training. Due to the costs incurred, I had to stop training and submit the project as is.

## Adding Audio as input prompt
The final piece of the multimodal assistant is integrating audio to the flow. The user can upload an image and ask a question related to the image and the model should be able to answer this. 

The first step for this process is to transcribe the audio input to text. We use the [WhisperX](https://github.com/m-bain/whisperX) model to do this processing. The transcribed text is then tokenized and passed to the LLM. When user uploads images and sends an audio as question, we process the audio using WhisperX model and translate it to text, which is used to calculate the text embeddings for the model.

# Huggingface space for Multimodal GPT 
The final step is to get the trained outputs from the model and create a huggingface app that uses gradio.
 [Here](https://huggingface.co/spaces/svaddi/MultimodalGPT) is the link to huggingface app. The image below shows a snapshot of the hugging face spaces app running

<img width="1578" alt="Screenshot 2024-02-28 at 6 26 20 PM" src="https://github.com/niharikavadapalli/ERA-V1/assets/135390352/87efb0f0-dfb8-4bdb-857a-a54d60430665">

# What more can be done
There can be some improvements that might improve the performance of the model. I did not got chance to experiment with the projection layer architecture that results in best performance. Although, I trained for about 30hrs it seems like the model need much more training to better learn the image details along with the question/answers. Also could have used OneCycleLR policy that could have helped the model learn more quickly saving a lot of time.

