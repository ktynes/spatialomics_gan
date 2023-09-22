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
        self.num_celltypes_used = np.random.randint(self.lb_celltypes, self.ub_celltypes)
        self.celltypes_idx = np.random.choice(self.num_celltypes, self.num_celltypes_used, replace=False)

        # Generate random cell type proportions
        proportions = np.random.dirichlet(np.ones(self.num_celltypes_used), 1)[0]
        all_proportions = np.array([proportions[i] if i in self.celltypes_idx else 0 for i in np.arange(self.num_celltypes)])

        # Use float32 for better memory usage
        self.dataset = torch.tensor(all_proportions, dtype=torch.float32)

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
        cell_types, celltype_counts = np.unique(celltype_labels, return_counts=True)
        cum_sum = np.insert(np.cumsum(celltype_counts), 0, 0)

        # Store data attributes
        self.num_cells = celltype_labels.size
        self.num_celltypes = cell_types.size
        self.cell_types = cell_types

        # Keep the data as sparse tensor for memory efficiency
        indices = torch.tensor(adata_singlecell.X.nonzero()).long()  # Convert to int64 tensor using .long()
        values = torch.tensor(adata_singlecell.X.data)
        shape = adata_singlecell.X.shape
        self.dataset = torch.sparse.FloatTensor(indices, values, shape)

        #self.dataset = torch.sparse.FloatTensor(torch.tensor(adata_singlecell.X.nonzero()), torch.tensor(adata_singlecell.X.data), adata_singlecell.X.shape)
        
        # Create a mask for cell types
        mask = np.zeros((self.num_celltypes, self.num_cells))
        for i in range(self.num_celltypes):
            mask[i, cum_sum[i]:cum_sum[i + 1]] = 1
        self.mask = torch.tensor(mask, requires_grad=False, dtype=torch.float32)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.adata_singlecell.X.shape[0]

class SpatialDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initialize the spatial dataset.
        
        Parameters:
            data_dir (str): Directory path containing the spatial data.
        """
        self.data_dir = data_dir
        adata_spatial = sc.read(data_dir)
        
        # Keep the data as sparse tensor for memory efficiency
        self.spatial_samples = torch.sparse.FloatTensor(torch.tensor(adata_spatial.X.nonzero()), torch.tensor(adata_spatial.X.data), adata_spatial.X.shape)

        # Use a numpy array for efficient data handling
        self.spatial_locations = np.column_stack((adata_spatial.obs['array_row'], adata_spatial.obs['array_col']))

    def __getitem__(self, index):
        return self.spatial_samples[index]

    def __len__(self):
        return self.spatial_samples.shape[0]