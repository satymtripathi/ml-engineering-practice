from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 Dataset wrapper.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: transforms.Compose = None,
    ):
        self.root = root
        self.train = train

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

        self.dataset = datasets.CIFAR10(
            root=self.root,
            train=self.train,
            download=True,
            transform=self.transform,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        return image, label
