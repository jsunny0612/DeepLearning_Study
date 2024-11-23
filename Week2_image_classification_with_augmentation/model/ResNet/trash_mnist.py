import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import gzip
import shutil
import requests

from torchvision.datasets.mnist import read_label_file
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
from torchvision.datasets.utils import download_url

log_dir = '/media/user/HDD2/sh/Chung_Ang/Week2_image_classification_with_augmentation/results/Resnet/mnist'
base_filename = 'output_log'
file_index = 1

while os.path.exists(f"{log_dir}/{base_filename}_{file_index}.txt"):
    file_index += 1

sys.stdout = open(f"{log_dir}/{base_filename}_{file_index}.txt", 'w')


def read_data_file(path, image=True):
    with gzip.open(path,'rb') as f:
        if image:
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        else:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


class CustomMNISTDataset(Dataset):
    urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ]

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    # self.model = MNIST()
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            image_file = os.path.join(self.root, "train-images-idx3-ubyte")
            label_file = os.path.join(self.root, "train-labels-idx1-ubyte")
        else:
            image_file = os.path.join(self.root, "t10k-images-idx3-ubyte")
            label_file = os.path.join(self.root, "t10k-labels-idx1-ubyte")

        # 파일이 존재하지 않으면 다운로드 후 압축 해제
        if not os.path.exists(image_file) or not os.path.exists(label_file):
            if download:
                for url in self.urls:
                    filename = url.split("/")[-1]
                    filepath = os.path.join(self.root, filename)

                    if not os.path.exists(filepath):
                        download_url(url, self.root, filename=filename)

                    with gzip.open(filepath, 'rb') as gz_file:
                        with open(filepath.replace(".gz", ""), 'wb') as out_file:
                            shutil.copyfileobj(gz_file, out_file)


            else:
                raise RuntimeError("Dataset not found. Use download=True to download it.")

        # 데이터를 로드합니다.
        self.images = read_data_file(image_file, image=True)
        self.labels = read_data_file(label_file, image=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Grayscale(3),  # Convert to 3 channels for VGG model
    transforms.Resize((224, 224)),  # Resize images to match VGG input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load training and test datasets
train_dataset = CustomMNISTDataset(root='/media/user/HDD2/sh/Chung_Ang/Week2_image_classification_with_augmentation/data/MNIST', train=True, transform=transform,download=True)
test_dataset = CustomMNISTDataset(root='/media/user/HDD2/sh/Chung_Ang/Week2_image_classification_with_augmentation/data/MNIST', train=False, transform=transform,download=True)

# DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the resnet18 model and modify the classifier for MNIST (10 classes)
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(in_features=num_ftrs, out_features=10)


# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Set up TensorBoard
writer = SummaryWriter(logdir='/media/user/SH/Chung_Ang/Week2_image_classification_with_augmentation/runs/mnist_experiment')

# Training function
def train(model, train_loader, criterion, optimizer, epoch):

    model.train()

    train_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_accuracy = 100. * correct_train / total_train

    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    return train_loss, train_accuracy


# Testing function
def test(model, test_loader, criterion, epoch):

    model.eval()

    test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accumulate loss and accuracy
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100. * correct_test / total_test

    # Log to TensorBoard
    # writer.add_scalars('Resnet_MNIST_Loss', {'Test': test_loss}, epoch + 1)
    # writer.add_scalars('Resnet_MNIST_Accuracy', {'Test': test_accuracy}, epoch + 1)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy

# Main training and testing loop
epochs = 10

total_start_time = time.time()

for epoch in range(epochs):

    print(f"\n======= Epoch {epoch + 1}/{epochs} =======")
    print(f'Epoch {epoch + 1}:')

    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = test(model, test_loader, criterion, epoch)

    writer.add_scalars('Resnet_MNIST_Loss', {'Train Loss': train_loss, 'Test Loss': test_loss}, epoch + 1)
    writer.add_scalars('Resnet_MNIST_Accuracy', {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}, epoch + 1)

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

hours, remainder = divmod(total_elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Close the TensorBoard writer
writer.close()

sys.stdout.close()
