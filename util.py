import torch
import wandb

def log_wandb_model(model, model_name, description, config, file_path):
    """
    Logs a PyTorch model to wandb.
    """
    model_artifact = wandb.Artifact(
        model_name, type="model",
        description=description,
        metadata=dict(config))

    torch.save(model.state_dict(), file_path)
    model_artifact.add_file(file_path)
    wandb.save(file_path)
    wandb.run.log_artifact(model_artifact)

def log_wandb_dataset(dataset, dataset_name, description, file_path):
    """
    Logs a PyTorch dataset to wandb.
    """
    data_artifact = wandb.Artifact(
        dataset_name, type="dataset",
        description=description,
        metadata={"type": dataset.type, "shape": dataset.shape})

    with data_artifact.new_file(dataset_name + ".pt", mode="wb") as file:
        torch.save(dataset, file)
            
    wandb.run.log_artifact(data_artifact)