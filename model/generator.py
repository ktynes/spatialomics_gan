from util import *
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):

    #Initialize generator module
    def __init__(self, num_cells, cell_types, mask):
        super(Generator, self).__init__()
        self.num_cells = num_cells
        self.num_celltypes = len(cell_types)
        self.mask = mask.clone().float().requires_grad_(False)

        self.linear_signatures = nn.Linear(self.num_cells, self.num_celltypes)


    #Implement forward pass of generator module
    def forward(self, random_proportions, singlecell_data):
        random_proportions = random_proportions.to(device)
        singlecell_data = singlecell_data.to(device)
        self.mask = self.mask.to(device)

        #print(singlecell_data.shape)
        #masked linear layer
        #print('mask shape', self.mask.shape)
        #print('mask type', self.mask.dtype)
        #print(self.mask)
        #print('weight shape',self.linear_signatures.weight.shape)
        #print('weight type', self.linear_signatures.weight.dtype)
        #print(self.linear_signatures.weight)
        self.linear_signatures.weight.data = self.linear_signatures.weight * self.mask #Check over this line for optimization error
        #print('weight2 shape',self.linear_signatures.weight.shape)
        #print('weight2 type', self.linear_signatures.weight.dtype)
        #print(self.linear_signatures.weight)
        #print(self.linear_signatures.weight.shape)
        #print(self.mask.shape)
        #print('singlecell shape', singlecell_data.T.shape)
        #print('singlecell type', singlecell_data.T.dtype)
        #print(singlecell_data.T)
        outs = self.linear_signatures(singlecell_data.T) #review this line
        #print('outs',outs.shape)
        #print('random_proportions',random_proportions.shape)

        #weighted average by random_proportions
        outs = torch.mm(random_proportions,outs.T)

        return outs