import torch
from torch import nn
import os
import numpy as np
import ot
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Evaluate Function
# =====================
def evaluate(generator, spatial_dataloader, singlecell_data, config, joined_name, results_dir):
    generator.eval()
    with torch.no_grad():
        celltypes_lb = config['celltypes_lb']
        celltypes_ub = config['celltypes_ub']
        num_epochs = config['num_epochs']

        spatial_dataset = spatial_dataloader.dataset
        spatial_size = len(spatial_dataset)
        real_spatial = spatial_dataset.spatial_samples

        num_celltypes = len(singlecell_data.cell_types)
        random_proportions = generate_random_proportions(spatial_size, celltypes_lb, celltypes_ub, num_celltypes)

        pseudo_spatial = generator(random_proportions.to(device), singlecell_data.dataset.to(device))
        save_path = os.path.join(results_dir, f'pseudo_spatial_{joined_name}_{num_epochs}epochs_{spatial_size}spots.npy')
        np.save(save_path, np.array(pseudo_spatial.cpu()))

        OT_Matrix = ot.dist(pseudo_spatial.cpu().numpy(), real_spatial.cpu().numpy(), metric='euclidean')
        a, b = np.ones((len(pseudo_spatial),)) / len(pseudo_spatial), np.ones((len(real_spatial),)) / len(real_spatial)
        dist = ot.emd2(a, b, OT_Matrix)
        
        print('Wasserstein_Distance: ', dist)
        wandb.log({"wasserstein_distance": dist})

# =====================
# Train Function
# =====================
def train(generator, discriminator, dataloader, singlecell_data, config, data_name):
    celltypes_lb = config['celltypes_lb']
    celltypes_ub = config['celltypes_ub']
    gen_lr = config['gen_lr']
    disc_lr = config['disc_lr']
    num_epochs = config['num_epochs']
    saved_weights_dir = config['saved_weights_dir']

    gen_optimizer = torch.optim.Adam(generator.parameters(), gen_lr)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), disc_lr)

    criterion = nn.BCELoss()

    wandb.watch(generator, criterion, log="all", log_freq=10)
    wandb.watch(discriminator, criterion, log="all", log_freq=10)

    for epoch in range(num_epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            generator.train()
            gen_optimizer.zero_grad()

            batch_size = len(batch)
            num_celltypes = len(singlecell_data.cell_types)
            random_proportions = generate_random_proportions(batch_size, celltypes_lb, celltypes_ub, num_celltypes).to(device)

            pseudo_spatial = generator(random_proportions, singlecell_data.dataset).to(device)
            real_spatial = batch.to(device)

            # Train the generator
            pseudo_discriminator_out = discriminator(pseudo_spatial.detach())
            generator_loss = criterion(pseudo_discriminator_out, torch.ones_like(pseudo_discriminator_out))
            generator_loss.backward()
            gen_optimizer.step()

            # Train the discriminator
            disc_optimizer.zero_grad()
            real_discriminator_out = discriminator(real_spatial)
            real_discriminator_loss = criterion(real_discriminator_out, torch.ones_like(real_discriminator_out))
            pseudo_discriminator_out = discriminator(pseudo_spatial.detach())
            pseudo_discriminator_loss = criterion(pseudo_discriminator_out, torch.zeros_like(pseudo_discriminator_out))
            discriminator_loss = (real_discriminator_loss + pseudo_discriminator_loss) / 2
            discriminator_loss.backward()
            disc_optimizer.step()

            wandb.log({
                "epoch": epoch, 
                "generator_loss": generator_loss.item(), 
                "discriminator_loss": discriminator_loss.item(), 
                "real_discriminator_loss": real_discriminator_loss.item(), 
                "pseudo_discriminator_loss": pseudo_discriminator_loss.item()
            })

    save_checkpoint(data_name, saved_weights_dir, generator, discriminator, gen_optimizer, disc_optimizer, num_epochs)

def generate_random_proportions(batch_size, celltypes_lb, celltypes_ub, num_celltypes):
    num_celltypes_used = np.random.randint(celltypes_lb, celltypes_ub, batch_size)
    celltypes_idx = [np.random.choice(num_celltypes, num_celltypes_used[i], replace=False) for i in range(batch_size)]
    proportions = [list(np.random.dirichlet(np.ones(num_celltypes_used[i]), 1)[0]) for i in range(batch_size)]
    batch_proportions = [[proportions[i].pop() if j in celltypes_idx[i] else 0 for j in np.arange(num_celltypes)] for i in range(batch_size)]
    return torch.tensor(batch_proportions).float()

def plot_loss_history(loss_history, path):
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
    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'gen_optimizer': gen_optimizer.state_dict(),
        'disc_optimizer': disc_optimizer.state_dict()
    }
    file_name = 'checkpoint_' + data_name + '_' + str(epoch_num) + 'epochs' + '.pth'
    save_path = os.path.join(saved_weights_dir, file_name)
    torch.save(state, save_path)