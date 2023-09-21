import torch
from torch import nn
import os
import json
import numpy as np
import csv
from scipy.stats import wasserstein_distance
import ot
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Evaluate Function
# =====================
def evaluate(generator, spatial_dataloader, singlecell_data, config, joined_name, results_dir):
    """
    Evaluates the performance of the model.
    """
    
    # Set the model to eval mode to avoid weights update
    generator.eval()
    with torch.no_grad():
        
        # Extract configurations
        celltypes_lb = config['celltypes_lb']
        celltypes_ub = config['celltypes_ub']
        num_epochs = config['num_epochs']
        
        spatial_dataset = spatial_dataloader.dataset
        spatial_size = len(spatial_dataset)
        real_spatial = spatial_dataset

        # Random cell type proportions for test data
        num_celltypes = len(singlecell_data.cell_types)
        random_proportions = generate_random_proportions(spatial_size, celltypes_lb, celltypes_ub, num_celltypes)

        # Generate pseudo and real spatial data
        pseudo_spatial = generator(random_proportions.to(device), singlecell_data.dataset.to(device))

        # Saving the pseudo spatial data
        save_path = os.path.join(results_dir, f'pseudo_spatial_{joined_name}_{num_epochs}epochs_{spatial_size}spots.npy')
        np.save(save_path, np.array(pseudo_spatial))

        # Compute OT Distance Metric
        OT_Matrix = ot.dist(pseudo_spatial, real_spatial, metric='euclidean')
        a = torch.tensor(ot.unif(len(pseudo_spatial)))
        b = torch.tensor(ot.unif(len(real_spatial)))
        dist = ot.emd2(a, b, OT_Matrix)
        
        print('Wasserstein_Distance: ', dist)
        wandb.log({"wasserstein_distance": dist})


# =====================
# Train Function
# =====================
def train(generator, discriminator, dataloader, singlecell_data, config, data_name):
    """
    Train the GAN model with provided generator and discriminator.
    """
    
    # Extract configurations
    celltypes_lb = config['celltypes_lb']
    celltypes_ub = config['celltypes_ub']
    gen_lr = config['gen_lr']
    disc_lr = config['disc_lr']
    num_epochs = config['num_epochs']
    saved_weights_dir = config['saved_weights_dir']

    # Optimizers
    gen_optimizer = torch.optim.Adam(generator.parameters(), gen_lr)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), disc_lr)

    # Loss Criterion
    criterion = nn.BCELoss()

    # Watch model gradients with wandb
    wandb.watch(generator, criterion, log="all", log_freq=10)
    wandb.watch(discriminator, criterion, log="all", log_freq=10)

    train_loss_history = []

    # Main training loop
    for epoch in range(num_epochs):
        gen_loss_history = []
        disc_loss_history = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            
            # Training mode
            generator.train()
            
            # Zero the gradients
            gen_optimizer.zero_grad()

            # Generate random cell type proportions
            batch_size = len(batch)
            num_celltypes = len(singlecell_data.cell_types)
            random_proportions = generate_random_proportions(batch_size, celltypes_lb, celltypes_ub, num_celltypes)

            # Generate pseudo and real spatial data
            pseudo_spatial = generator(random_proportions, singlecell_data.dataset).to(device)
            pseudo_labels = torch.zeros((batch_size, 1))
            real_spatial = batch.to(device)
            real_labels = torch.ones((batch_size, 1))

            # Train the generator
            pseudo_discriminator_out = discriminator(pseudo_spatial)
            generator_loss = criterion(pseudo_discriminator_out, real_labels)
            gen_loss_history.append(generator_loss.item())
            generator_loss.backward(retain_graph=True)

            # Train the discriminator
            disc_optimizer.zero_grad()
            real_discriminator_out = discriminator(real_spatial)
            real_discriminator_loss = criterion(real_discriminator_out, real_labels)
            pseudo_discriminator_out = discriminator(pseudo_spatial)
            pseudo_discriminator_loss = criterion(pseudo_discriminator_out, pseudo_labels)
            discriminator_loss = (real_discriminator_loss + pseudo_discriminator_loss) / 2
            disc_loss_history.append(discriminator_loss.item())
            discriminator_loss.backward(retain_graph=True)
            
            # Step the optimizers
            gen_optimizer.step()
            disc_optimizer.step()
            
            # Logging
            wandb.log({
                "epoch": epoch, 
                "generator_loss": generator_loss, 
                "discriminator_loss": discriminator_loss, 
                "real_discriminator_loss": real_discriminator_loss, 
                "pseudo_discriminator_loss": pseudo_discriminator_loss
            })

        train_loss_history.append((gen_loss_history, disc_loss_history))

    # Save the models and plot loss
    save_checkpoint(data_name, saved_weights_dir, generator, discriminator, gen_optimizer, disc_optimizer, num_epochs)
    plot_loss_history(gen_loss_history, f'results/{data_name}/gen_loss_{num_epochs}.png')
    plot_loss_history(disc_loss_history, f'results/{data_name}/disc_loss_{num_epochs}.png')

    # # Save & Plot loss
    # path = os.path.join('results',data_name)
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # with open(os.path.join(path, "loss_history" + str(num_epochs)+".csv"),"a") as f:
    #         wr = csv.writer(f)
    #         wr.writerows([["gen_loss", "disc_loss"]])
    #         for losses in train_loss_history:
    #             for batch_loss in zip(losses[0], losses[1]):
    #                 wr.writerows([batch_loss])

    # loss_path_gen = os.path.join(path,'gen_loss_' + str(num_epochs) +'.png')
    # loss_path_disc = os.path.join(path,'disc_loss_' + str(num_epochs) +'.png')
    # plot_loss_history(gen_loss_history, loss_path_gen)
    # plot_loss_history(disc_loss_history, loss_path_disc)

def generate_random_proportions(batch_size, celltypes_lb, celltypes_ub, num_celltypes):
    #Cell Types
    num_celltypes_used = np.random.randint(celltypes_lb,celltypes_ub, batch_size)
    celltypes_idx = [np.random.choice(num_celltypes, num_celltypes_used[i], replace=False) for i in range(batch_size)]

    #Cell Type Proportions
    proportions = [list(np.random.dirichlet(np.ones(num_celltypes_used[i]), 1)[0]) for i in range(batch_size)] #proportions for celltypes used
    batch_proportions = [[proportions[i].pop() if j in celltypes_idx[i] else 0 for j in np.arange(num_celltypes)] for i in range(batch_size)] #adds zeroes
    batch_proportions = torch.tensor(batch_proportions).float()

    return batch_proportions

def plot_loss_history(loss_history, path):
    """Plots the loss history"""
    plt.figure()
    epoch_idxs = range(len(loss_history))

    plt.plot(epoch_idxs, loss_history, "-b")
    plt.title("Loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.xticks(np.arange(0, max(epoch_idxs)+1, step=1))
    plt.savefig(path)

def save_checkpoint(data_name, saved_weights_dir, generator, discriminator, gen_optimizer, disc_optimizer, epoch_num):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param generator: generator model
    :param discriminator: discriminator model
    :param gen_optimizer: optimizer to update generator's weights
    :param disc_optimizer: optimizer to update discriminator's weights
    :param epoch: epoch number
    """
    state = {'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'disc_optimizer': disc_optimizer.state_dict()}
    file_name = 'checkpoint_' + data_name + '_' + str(epoch_num) + 'epochs' + '.pth'
    save_path = os.path.join(saved_weights_dir, file_name)
    #filename = 'saved_weights/checkpoint_' + data_name + '_' + str(epoch_num) + '.pth'

    saved_weights_dir = './saved_weights/'

    torch.save(state, save_path)
    # try:
    #     torch.save(state, filename, _use_new_zipfile_serialization = True)
    # except:
    #     torch.save(state, filename, _use_new_zipfile_serialization = False)
    

#     cell_type_proportions = torch.randint(0, 1, size=(batch_size, input_length)).float()

# # Random number of cells
# N_cells = torch.randInt(N_lb, N_ub, size = (batch_size))
# N_cells_by_type = torch.round(N_cells*cell_type_proportions)
# corr_proportions = N_cells_by_type/N_cells