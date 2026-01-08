import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BinaryImageDataset(Dataset):
    """
    Generic binary image dataset.
    Folder structure:
    root/
        class0/
        class1/
    """

    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose = None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = self._load_samples()
        self._sanity_check_metadata()

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Scan folders and collect (image_path, label).
        """
        samples = []
        class_to_label = {}

        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            class_to_label[class_name] = idx

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_path, file_name)
                    samples.append((img_path, idx))

        return samples

    def _sanity_check_metadata(self):
        assert len(self.samples) > 0, "Dataset is empty"

        labels = [label for _, label in self.samples]
        assert set(labels) == {0, 1}, "Expected binary labels {0,1}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image, label
