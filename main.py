from data.dataset import BinaryImageDataset
from utils.sanity_checks import check_image_tensor, check_batch
from torch.utils.data import DataLoader


def main():
    dataset = BinaryImageDataset(
        root_dir=r"C:\Users\satyam.tripathi\Desktop\Learning_Project\Data"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )
    images, labels = next(iter(dataloader))

    # Sanity check on the batch
    check_batch(images, labels)

    print("Batch image shape:", images.shape)
    print("Batch labels:", labels)


if __name__ == "__main__":
    main()
