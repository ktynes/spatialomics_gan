from util import *
import torch
import torch.nn as nn

class Generator2(nn.Module):

    #Initialize generator module
    def __init__(self, num_cells, cell_types, mask, embed_dim):
        super(Generator2, self).__init__()
        self.num_cells = num_cells
        print(cell_types)
        self.num_celltypes = len(cell_types)
        print(self.num_celltypes)
        self.mask = mask.clone().float().requires_grad_(False)
        self.embed_dim = embed_dim

        self.linear_signatures = nn.Linear(self.num_cells, self.num_celltypes)
        self.fc1= nn.Linear(self.num_celltypes, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.num_celltypes)
        self.activation = nn.ReLU()

    #Implement forward pass of generator module
    def forward(self, random_proportions, singlecell_data):
        random_proportions = random_proportions.to(device)
        singlecell_data = singlecell_data.to(device)
        self.mask = self.mask.to(device)
        
        #masked linear layer
        self.linear_signatures.weight.data = self.linear_signatures.weight * self.mask #Check over this line for optimization error
        #print('singlecell_data: ', singlecell_data.type)
        outs = self.linear_signatures(singlecell_data.T) #review this line
        outs = self.fc1(outs)
        outs = self.activation(outs)
        outs = self.fc2(outs)
        #weighted average by random_proportions
        #print('outs: ', outs.type)
        outs = torch.mm(random_proportions,outs.T)

        return outs