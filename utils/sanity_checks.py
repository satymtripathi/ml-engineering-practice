import torch


def check_image_tensor(image: torch.Tensor):
    """
    Sanity checks for image tensor.
    """
    assert isinstance(image, torch.Tensor), "Image is not a torch Tensor"
    assert image.ndim == 3, f"Expected 3D tensor (C,H,W), got {image.ndim}D"
    assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"
    assert image.dtype == torch.float32, f"Expected float32, got {image.dtype}"
    assert image.min() >= 0.0 and image.max() <= 1.0, "Image values out of range [0,1]"


def check_binary_label(label):
    """
    Sanity check for binary classification label.
    """
    assert isinstance(label, int), "Label must be an integer"
    assert label in (0, 1), f"Invalid label {label}, expected 0 or 1"


def check_sample(image: torch.Tensor, label):
    """
    Combined sanity check for a single sample.
    """
    check_image_tensor(image)
    check_binary_label(label)

def check_batch(images: torch.Tensor, labels: torch.Tensor):
    """
    Sanity checks for a batch of images and labels.
    """

    # ---- images checks ----
    assert isinstance(images, torch.Tensor), "Images batch is not a torch Tensor"
    assert images.ndim == 4, f"Expected 4D tensor (B,C,H,W), got {images.ndim}D"
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
    assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
    assert images.min() >= 0.0 and images.max() <= 1.0, "Image values out of range [0,1]"

    # ---- labels checks ----
    assert isinstance(labels, torch.Tensor), "Labels batch is not a torch Tensor"
    assert labels.ndim == 1, f"Expected 1D labels tensor, got {labels.ndim}D"
    assert labels.shape[0] == images.shape[0], "Mismatch between images and labels batch size"

    unique_labels = set(labels.tolist())
    assert unique_labels.issubset({0, 1}), f"Invalid labels found: {unique_labels}"

