import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from random import random, shuffle
from PIL import Image
import time
import os


class MinimapArrowDataset(Dataset):

    def __init__(self, labels, train):
        super().__init__()
        self.labels = labels
        self.train = train
        self.transform_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path, facing = self.labels[index]
        img = Image.open(img_path)

        facing = torch.tensor(facing, dtype=float)
        radius = 1
        y = torch.sin(facing).mul(radius)
        x = torch.cos(facing).mul(radius)

        # Horizontal flip
        if self.train and random() < 0.25:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x = x.mul(-1)

        # Vertical flip
        if self.train and random() < 0.25:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            y = y.mul(-1)

        return self.transform_to_tensor(img), x, y


class DistanceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_pred: torch.Tensor, y_pred: torch.Tensor, x_actual: torch.Tensor, y_actual: torch.Tensor):
        return torch.sqrt((x_pred-x_actual).pow(2) + (y_pred-y_actual).pow(2))


def load_labels(file_name):
    file = open(file_name)
    images = []
    for line in file.readlines():
        fname, facing = line.split(",")
        images.append((fname, float(facing)))
    file.close()
    return images


def get_network():
    return nn.Sequential(
        nn.BatchNorm2d(3),

        nn.Conv2d(3, 32, 3),
        nn.ReLU(),

        nn.MaxPool2d(2, 2),

        nn.Conv2d(32, 32, 3),
        nn.ReLU(),

        nn.MaxPool2d(2, 2),

        nn.Conv2d(32, 32, 3),
        nn.ReLU(),

        nn.MaxPool2d(2, 2),

        nn.Flatten(),

        nn.Linear(128, 128),
        nn.ReLU(),

        nn.Dropout(0.2),

        nn.Linear(128, 64),
        nn.ReLU(),

        nn.Dropout(0.1),

        nn.Linear(64, 2),
        # nn.Tanh()
    )


def train():
    TS = time.strftime("%Y%m%d_%H%M%S")

    device = "cuda"
    num_epochs = 20000
    batch_size = 64
    output_dir = Path(__file__).parent / f"train_{TS}"
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = Path(__file__).parent / "dataset"
    labels = load_labels(dataset_dir / "labels.csv")
    labels = [(dataset_dir / "images" / file_name, j)
              for (file_name, j) in labels]
    shuffle(labels)

    split_index = int(0.8 * len(labels))

    train_labels = labels[:split_index]
    val_labels = labels[split_index:]

    train_dataset = MinimapArrowDataset(train=True, labels=train_labels)
    val_dataset = MinimapArrowDataset(val_labels, train=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        persistent_workers=True
    )

    model = get_network().to(device)
    loss_fn = DistanceLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0.0005)

    print(sum(p.numel() for p in model.parameters()))

    min_loss = float('inf')

    def write_to_csv(train_loss, val_loss):
        csv = open(output_dir / "stats.csv", "a+")
        csv.write(f"{train_loss},{val_loss}\n")
        csv.close()

    write_to_csv("train", "validation")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for (images, x_actual, y_actual) in train_dataloader:
            optimizer.zero_grad()
            images = images.to(device)
            x_actual = x_actual.to(device)
            y_actual = y_actual.to(device)

            output = model(images)
            loss = loss_fn(output[:, 0], output[:, 1],
                           x_actual, y_actual).sum()
            train_loss += loss.item()

            (loss/images.size()[0]).backward()

            optimizer.step()
        # print("Train loss:", loss_sum)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            for (images, x_actual, y_actual) in train_dataloader:
                images = images.to(device)
                x_actual = x_actual.to(device)
                y_actual = y_actual.to(device)
                output = model(images)
                loss = loss_fn(output[:, 0], output[:, 1], x_actual, y_actual)
                val_loss += loss.sum().item()

        if val_loss < min_loss:
            torch.jit.script(model).save(output_dir / "best.pt")
            print(epoch, val_loss)
            min_loss = val_loss

        write_to_csv(train_loss, val_loss)


if __name__ == "__main__":
    train()
