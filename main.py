from util import *
import os
import torch
import argparse
import json
from pathlib import Path
import re
import numpy as np
from torch.utils.data import DataLoader
import wandb
import yaml

from preprocessing import SingleCellDataset, SpatialDataset
from train_eval import train, evaluate
from model.discriminator import Discriminator
from model.generator import Generator
from model.generator2 import Generator2

# Setting the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

if __name__ == '__main__':
    
    # SET RANDOM SEED
    np.random.seed(33333)

    # Load experiment configurations from yaml file
    with open(os.path.join('./configs', 'default.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    # Construct data paths
    data_name = config['spatial_sample'] + "_" + Path(config['singlecell_file']).stem
    singlecell_path = os.path.join(config['data_dir'], 'Aligned Data', 'adata_singlecell_cor_unnorm.loom')
    spatial_path = os.path.join(config['data_dir'], 'Aligned Data', 'adata_spatial_cor_unnorm.loom')

    # Results directory
    joined_name = data_name + str(config['exp_num'])
    results_dir = os.path.join(config['results_dir'], joined_name)

    # Initialize wandb for experiment tracking
    with wandb.init(project="testing", entity="kdtynes", config=config):
        
        config = wandb.config

        # Load single-cell data
        singlecell_dataset = SingleCellDataset(singlecell_path)
        num_cells = singlecell_dataset.num_cells
        cell_types = singlecell_dataset.cell_types
        num_genes = singlecell_dataset.num_genes
        mask = singlecell_dataset.mask

        # Load spatial data and set DataLoader
        spatial_dataset = SpatialDataset(spatial_path)
        spatial_dataloader = DataLoader(spatial_dataset, batch_size=config['batch_size'], shuffle=True)

        # Model Initialization
        if not config['checkpoint_path']:
            # Initialize new generator and discriminator
            generator = Generator2(num_cells, cell_types, mask, config['gen_embed_dim']).to(device)
            discriminator = Discriminator(num_genes, config['embed_dim']).to(device)
        else:
            # Load model from checkpoint
            print("STARTING FROM CHECKPOINT")
            checkpoint = torch.load(config['checkpoint_path'])
            generator = checkpoint['generator']
            discriminator = checkpoint['discriminator']
            gen_optimizer = checkpoint['gen_optimizer']
            disc_optimizer = checkpoint['disc_optimizer']

            # Overwrite the number of epochs from the checkpoint, if needed
            config['num_epochs'] = checkpoint['num_epochs']

        # Training Phase
        if config['train_bool']:
            print("TRAINING NEW MODEL")
            train(generator, discriminator, spatial_dataloader, singlecell_dataset, config, joined_name)
        
        # Evaluation Phase (can be done after training or directly if only evaluating)
        evaluate(generator, spatial_dataset, singlecell_dataset, config, joined_name, results_dir)

        # PSEUDOCODE: Additional Logging
        # log_wandb_model(generator, generator_name, generator_description, config, file_path)
        # TODO: Add functions for more detailed logging to wandb, visualizations, and model metrics