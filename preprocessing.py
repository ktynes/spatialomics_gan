'''
Preprocess Data
'''

from util import *
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset

class RandomCellProportions(Dataset):
    def __init__(self, lb_celltypes, ub_celltypes, num_celltypes):
        """
        Initialize dataset to randomly generate cell type proportions.
        
        Parameters:
            lb_celltypes (int): Lower bound for number of cell types.
            ub_celltypes (int): Upper bound for number of cell types.
            num_celltypes (int): Total number of cell types available.
        """
        self.lb_celltypes = lb_celltypes
        self.ub_celltypes = ub_celltypes
        self.num_celltypes = num_celltypes

        # Randomly select number of cell types used
        self.num_celltypes_used = np.random.randint(self.lb_celltypes,self.ub_celltypes)
        self.celltypes_idx = np.random.choice(self.num_celltypes, self.num_celltypes_used, replace=False)

        # Generate random cell type proportions
        proportions = list(np.random.dirichlet(np.ones(self.num_celltypes_used), 1)[0])
        all_proportions = [proportions.pop() if i in self.celltypes_idx else 0 for i in np.arange(self.num_celltypes)] 

        self.dataset = all_proportions

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.ub_celltypes

class SingleCellDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initialize the single cell RNA dataset.
        
        Parameters:
            data_dir (str): Directory path containing the scRNA-seq data.
        """
        self.data_dir = data_dir
        adata_singlecell = sc.read(self.data_dir)
        
        # Sort data based on 'Class' column
        idx = np.argsort(adata_singlecell.obs.Class)
        self.adata_singlecell = adata_singlecell[idx]

        # Extract unique cell type labels and their counts
        celltype_labels = adata_singlecell.obs['Class']
        cell_types, celltype_counts = np.unique(celltype_labels, return_counts = True)
        cum_sum = np.insert(np.cumsum(celltype_counts), 0, 0)

        # Store data attributes
        self.num_cells = celltype_labels.size
        self.num_celltypes = cell_types.size
        self.cell_types = cell_types
        self.dataset = torch.tensor(adata_singlecell.X.todense())
        self.num_genes = len(self.dataset[0])

        # Create a mask for cell types
        mask = np.zeros((self.num_celltypes, self.num_cells))
        for i in range(self.num_celltypes):
            mask[i, cum_sum[i]:cum_sum[i + 1]] = 1
        self.mask = torch.tensor(mask, requires_grad=False)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return np.shape(self.adata_singlecell.X)[0]

class SpatialDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initialize the spatial dataset.
        
        Parameters:
            data_dir (str): Directory path containing the spatial data.
        """
        self.data_dir = data_dir
        adata_spatial = sc.read(data_dir)
        
        # Extract and store spatial samples and locations
        self.spatial_samples = torch.tensor(adata_spatial.X.todense())
        self.spatial_locations = list(zip(adata_spatial.obs['array_row'], adata_spatial.obs['array_col']))

    def __getitem__(self, index):
        return self.spatial_samples[index]

    def __len__(self):
        return len(self.spatial_samples)