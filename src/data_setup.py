"""Creates PyTorch DataLoaders for image classification data."""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates and returns (train_dataloader, test_dataloader, class_names)."""
  if torch.cuda.is_available():
      device = "cuda"
      pin_memory = True
  elif torch.backends.mps.is_available():
      device = "mps"
      pin_memory = False
  else:
      device = "cpu"
      pin_memory = False
  
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  class_names = train_data.classes

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=pin_memory,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=pin_memory,
  )

  return train_dataloader, test_dataloader, class_names
