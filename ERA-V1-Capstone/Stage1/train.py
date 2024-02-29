import os
import torch
import numpy as np
from transformers import AutoTokenizer, CLIPVisionModel, AutoProcessor
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import Callback
import wandb
import json

from dataset import CocoImagesAndCaptionsDataset
from model import CaptionGeneratorGPT


class SaveCheckpoints(ModelCheckpoint):
    def __init__(self, checkpoint_save_dir, save_freq, verbose: bool = False):
        super().__init__()
        self.save_dir = checkpoint_save_dir
        self.save_freq = save_freq
        self.verbose = verbose

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.save_freq == 0:
            projection_layer_filename = os.path.join(self.save_dir, f"projection_layer_ckpt_{trainer.global_step}.pt")
            resnet_layer_filename = os.path.join(self.save_dir, f"resnet_layer_ckpt_{trainer.global_step}.pt")
            torch.save(trainer.model.projection_layer.state_dict(), projection_layer_filename)
            torch.save(trainer.model.resnet_layer.state_dict(), resnet_layer_filename)
    
    def on_validation_end(self, trainer, pl_module):
        pass


class PrintAccuracyAndLoss(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        train_loss = trainer.callback_metrics['train_loss']
        trainer.model.log("train_step_loss", train_loss)
        if batch_idx % 500 == 0:
            print(f"Step: {trainer.global_step}: train_loss={train_loss:.4f}")


def train_model(model, train_dataloader, val_dataloader, logger=None, ckpt_path=None, max_training_steps=2):
    trainer = Trainer(
        enable_checkpointing=True,
        max_steps=max_training_steps,
        max_epochs = 3,
        accelerator="auto",
        devices = 1, 
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=10),
                   SaveCheckpoints(check_point_save_dir, save_freq, verbose=True),
                   PrintAccuracyAndLoss()],
        num_sanity_val_steps=0,
        val_check_interval = val_check_interval,
        precision="16"
    )
    
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    return trainer

def get_captions_and_images(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    captions = {}
    image_paths = {}
    image_ids = []
    annotations = data['annotations']
    images = data['images']

    for img in images:
        image_paths[img['id']] = img['coco_url']
        image_ids.append(img['id'])
    
    for annotation in annotations:
        captions[annotation['image_id']] = annotation['caption']
    
    print(f"total image ids: {len(image_paths)}, total images: {len(image_paths)}, total captions: {len(captions)}")
    return captions, image_paths, image_ids

if __name__ == '__main__':

    # Define configs
    annFile='annotations/captions_train2017.json'
    #annFile='annotations/captions_val2017.json'   
    
    val_split_size = 0.001
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_embed_size = 768
    llm_embed_size = 2560
    max_training_steps = 100000
    seq_len = 64
    log_dir = 'phi_pretrain'
    exp_name = 'phi2_proj_layer'
    check_point_save_dir = 'phi2_projection_checkpoints/'
    os.makedirs(check_point_save_dir,exist_ok = True)
    save_freq = 1000
    val_check_interval = 1000
    lr = 1e-4
    log_dir = './logs'
    resume_training=False
    proj_weights_path = 'phi2_projection_checkpoints/ckpt_latest.pt'


    # Instantiate WandB logger
    wandb.login()
    wandb_logger = WandbLogger(save_dir=log_dir, name='mm_phi_stage1')
    text_table = wandb.Table(columns=["training_step", "loss", "text"])
    val_data_log = []

    torch.set_float32_matmul_precision('medium')

    # Define the models and tokenizers
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_preprocessor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")


    # Define dataset and dataloaders
    captions, raw_images_list, image_ids_list = get_captions_and_images(annFile)

    keys_list = np.array(image_ids_list)
    rand_indices = np.arange(len(keys_list))
    np.random.shuffle(rand_indices)

    val_split = int(len(keys_list)*val_split_size)

    val_image_ids, train_image_ids = keys_list[rand_indices[:val_split]], keys_list[rand_indices[val_split:]] 

    print(f"Train dataset size: {len(train_image_ids)}")
    print(f"Valid dataset size: {len(val_image_ids)}")


    train_ds = CocoImagesAndCaptionsDataset(train_image_ids, 
                                raw_images_list, 
                                captions, 
                                clip_model,
                                clip_preprocessor,
                                phi_tokenizer,
                                seq_len=seq_len)

    val_ds = CocoImagesAndCaptionsDataset(val_image_ids,  
                                raw_images_list, 
                                captions, 
                                clip_model,
                                clip_preprocessor,
                                phi_tokenizer,
                                seq_len=seq_len)

    train_dataloader = DataLoader(dataset = train_ds,
                                batch_size = batch_size,
                                num_workers = 3,
                                collate_fn = None,
                                shuffle = True)
    val_dataloader = DataLoader(dataset = val_ds,
                                batch_size = batch_size,
                                num_workers = 3,
                                collate_fn = None,
                                shuffle = True)


    # Define and train model
    captionGeneratorGPT = CaptionGeneratorGPT(clip_embed_size,
                                            llm_embed_size,)
    optimizer = torch.optim.Adam(captionGeneratorGPT.parameters(), lr=1.0e-4, eps=1e-9)
    captionGeneratorGPT.set_optimizer(optimizer)

    trainer = train_model(captionGeneratorGPT, train_dataloader, val_dataloader, logger = wandb_logger, max_training_steps=max_training_steps)