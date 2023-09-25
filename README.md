# `spatialomics_gan`: A Comprehensive Approach to Spatial Transcriptomics Data

## Project Description
The `spatialomics_gan` project endeavors to harness the intricate nature of spatial transcriptomics (ST) data through computational mechanisms. Using the robust capabilities of Generative Adversarial Networks (GANs) combined with intricately designed linear layers, the primary objective is to effectively represent and interpret gene expression patterns within a spatial perspective. This initiative bridges crucial understanding gaps in spatial transcriptomics.

## Repository Structure
This repository has been organized for intuitive navigation and understanding:
- **configs**: Configuration files crucial for various experiment setups.
- **model**: Models that form the backbone of this project.
- **main.py**: The core script to commence both training and evaluation processes.
- **preprocessing.py**: This script is essential for precise data preprocessing.
- **train_eval.py**: Functions dedicated to the model's training and evaluation are located here.
- **util.py**: Utility functions designed to facilitate various tasks within the project are stored here.

## Model Architecture: Spatial-Omics GAN

![Spatial-Omics GAN Framework]("spatialomics_gan_framework.png")
Spatial-Omics Generative Adversarial Network Framework: This illustration showcases the GAN's pseudo-spot generation process. Drawing from single-cell data, the network undertakes a randomized cell type assignment to craft synthetic spatial-omics spots. These generated spots are then juxtaposed against authentic spots, facilitating a discriminator-guided classification. The visualization underscores the network's capability to mimic true spatial patterns through its generative processes.

### Overview
The Spatial-Omics GAN, crafted through diligent design phases, is adept at processing and interpreting spatial transcriptomics data, especially focusing on tissue-specific information. It uniquely incorporates the Dirichlet distribution to handle complexities unique to spatial transcriptomics datasets.

### Spot-Level Design
The design centers on the premise that each spot is an intersection of various cell types, each boasting its distinct gene expression signature. The Generative component of the GAN capitalizes on this knowledge, converting random noise into identifiable pseudo-spots. These pseudo-spots are then correlated with random cell-type assignments influenced by single-cell data.

## Configurations and Parameters
The provided configurations ensure flexibility for users to modify the model as needed:
- **checkpoint_path**: Directs to any existing model checkpoint. Default: `False`
- **train_bool**: Indicates the model's mode of operation. Default: `True`
- **gen_lr**: Generator Learning Rate. Default: `0.003`
- **disc_lr**: Discriminator Learning Rate. Default: `0.003`
- **embed_dim**: Discriminator Embedding Dimension. Default: `256`
- **gen_embed_dim**: Generator Embedding Dimension. Default: `256`
- **num_epochs**: Specifies the number of training epochs. Default: `5`
... [additional parameters]
For a detailed list and more insights, please [refer to the config file]("configs/default.yaml").

## Prerequisites
For replication or further advancements:
- Install necessary libraries using `pip install -r requirements.txt`.
- Use of a virtual environment for dependency management is strongly advised.
- Ensure GPU support for efficient model training and evaluation.

## Model Usage
### Data Preparation
Process any acquired spatial transcriptomics data with `preprocessing.py`.

### Model Configuration
Fine-tune parameters within `configs` to meet specific research needs.

### Training
Initiate model training using:

```bash
python main.py --config=configs/default.yaml
```

### Evaluation
Evaluate the model's performance using:

```bash
python train_eval.py [ADDITIONAL PARAMETERS IF NEEDED]
```
Kevin Tynes, PhD student in Machine Learning, Computational Science & Engineering, Georgia Institute of Technology
