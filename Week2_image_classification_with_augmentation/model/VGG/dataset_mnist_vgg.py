import torch
import time
import sys
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from Week2_image_classification_with_augmentation.model.VGG.vgg16 import vgg16
from tensorboardX import SummaryWriter

log_dir = '/media/user/HDD2/sh/Chung_Ang/Week2_image_classification_with_augmentation/results/test'
base_filename = 'output_log'
file_index = 1

while os.path.exists(f"{log_dir}/{base_filename}_{file_index}.txt"):
    file_index += 1

sys.stdout = open(f"{log_dir}/{base_filename}_{file_index}.txt", 'w')


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

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.image_file = os.path.join(self.root, "train-images-idx3-ubyte")
            self.label_file = os.path.join(self.root, "train-labels-idx1-ubyte")
        else:
            self.image_file = os.path.join(self.root, "t10k-images-idx3-ubyte")
            self.label_file = os.path.join(self.root, "t10k-labels-idx1-ubyte")

        if not os.path.exists(self.image_file) or not os.path.exists(self.label_file):
            print("MNIST Datasets are not existed.")

        self.images = self.read_data_file(self.image_file, image=True)
        self.labels = self.read_data_file(self.label_file, image=False)

    def read_data_file(self, path, image=True):
        with open(path, 'rb') as f:
            if image:
                data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            else:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

        return data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for VGG model
    transforms.Resize((224, 224)),  # Resize images to match VGG input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load training and test datasets
train_dataset = CustomMNISTDataset(root='/media/user/HDD2/sh/Chung_Ang/Week2_image_classification_with_augmentation/data/MNIST/raw', train=True, transform=transform)
test_dataset = CustomMNISTDataset(root='/media/user/HDD2/sh/Chung_Ang/Week2_image_classification_with_augmentation/data/MNIST/raw', train=False, transform=transform)

# Split the train dataset into train and validation sets
train_size = 55000
val_size = 5000
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# DataLoader for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the resnet18 model and modify the classifier for MNIST (10 classes)
model = vgg16(num_classes=10)

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

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Accumulate loss and accuracy
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct_val / total_val

    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    return val_loss, val_accuracy

# Testing function
def test(model, test_loader, criterion):
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

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy

# Main training and testing loop
epochs = 10

total_start_time = time.time()

for epoch in range(epochs):
    print(f"\n======= Epoch {epoch + 1}/{epochs} =======")
    print(f'Epoch {epoch + 1}:')

    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    val_loss, val_acc = validate(model, val_loader, criterion)
    test_loss, test_acc = test(model, test_loader, criterion)

    writer.add_scalars('VGG_MNIST_Loss', {'Train Loss': train_loss, 'Validation Loss': val_loss, 'Test Loss': test_loss}, epoch + 1)
    writer.add_scalars('VGG_MNIST_Accuracy', {'Train Accuracy': train_acc, 'Validation Accuracy': val_acc, 'Test Accuracy': test_acc}, epoch + 1)

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

hours, remainder = divmod(total_elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Close the TensorBoard writer
writer.close()
sys.stdout.close()
