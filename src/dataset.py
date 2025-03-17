from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image


class FilmDataset(Dataset):
    def __init__(self, dir: Path, res=512):
        self.dir = dir
        self.files = []
        for file in dir.glob("**/*.jpg"):
            self.files.append(file)

        self.trans_both = T.Compose([
            T.RandomResizedCrop(res, scale=(0.3, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ])
        self.trans_x = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.RandomSolarize(threshold=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2.0),
        ])
        self.trans_y = T.Compose([
            T.ElasticTransform(alpha=10.0),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_img = read_image(str(self.files[idx])).float() / 255.0
        y_img = torch.zeros((1, x_img.shape[1], x_img.shape[2]), dtype=torch.float32)

        # Draw rect label.
        with open(self.files[idx].with_suffix(".txt"), "r") as f:
            min_x, min_y, max_x, max_y = map(int, f.readline().split())
        y_img[0, min_y:max_y, min_x:max_x] = 1.0

        # Apply augs
        both = torch.cat([x_img, y_img], dim=0)
        both = self.trans_both(both)
        x_img, y_img = both[:-1], both[-1:]
        x_img = self.trans_x(x_img)
        y_img = self.trans_y(y_img)

        return x_img, y_img


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    dataset = FilmDataset(args.data)

    # Draw 3 by 6 plot of X Y samples.
    indices = random.sample(range(len(dataset)), 6)
    fig, axs = plt.subplots(3, 6, dpi=300)
    axs[0, 0].set_title("X")
    axs[1, 0].set_title("Y")
    axs[2, 0].set_title("X + Y")
    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        axs[0, i].imshow(x[0], cmap="gray")
        axs[1, i].imshow(y[0], cmap="gray")
        x[y > 0.5] = 1
        axs[2, i].imshow(x[0], cmap="gray")
        for j in range(3):
            axs[j, i].axis("off")

    x, y = dataset[0]
    print("X", x.shape, x.min(), x.max())
    print("Y", y.shape, y.min(), y.max())

    #plt.show()
    plt.savefig("dataset.jpg")
