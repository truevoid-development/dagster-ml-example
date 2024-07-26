import os
from typing import Dict, Tuple

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# Constants
DATA_ROOT = "~/data"
IMAGE_SIZE = 28
NUM_CLASSES = 10
HIDDEN_SIZE = 512

def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare and return train and test dataloaders for FashionMNIST dataset.

    Args:
        batch_size (int): The batch size for the dataloaders.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test dataloaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    with FileLock(os.path.expanduser(f"{DATA_ROOT}.lock")):
        training_data = datasets.FashionMNIST(
            root=DATA_ROOT, train=True, download=True, transform=transform
        )
        test_data = datasets.FashionMNIST(
            root=DATA_ROOT, train=False, download=True, transform=transform
        )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    """Neural network model for FashionMNIST classification."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_func_per_worker(config: Dict[str, float]) -> None:
    """
    Training function to be executed on each worker.

    Args:
        config (Dict[str, float]): Configuration dictionary containing
            learning rate, number of epochs, and batch size per worker.
    """
    learning_rate = config["lr"]
    num_epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)

    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    model = ray.train.torch.prepare_model(NeuralNetwork())

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        if ray.train.get_context().get_world_size() > 1:
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for inputs, targets in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)

                test_loss += loss.item()
                num_total += targets.shape[0]
                num_correct += (predictions.argmax(1) == targets).sum().item()

        avg_test_loss = test_loss / len(test_dataloader)
        accuracy = num_correct / num_total

        ray.train.report(metrics={"loss": avg_test_loss, "accuracy": accuracy})


def train_fashion_mnist(num_workers: int = 2, use_gpu: bool = False) -> None:
    """
    Train the FashionMNIST model using distributed training.

    Args:
        num_workers (int): Number of workers for distributed training.
        use_gpu (bool): Whether to use GPU for training.
    """
    global_batch_size = 32

    train_config = {
        "lr": 1e-3,
        "epochs": 10,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    result = trainer.fit()
    print(f"Training result: {result}")


if __name__ == "__main__":
    train_fashion_mnist(num_workers=2, use_gpu=False)
