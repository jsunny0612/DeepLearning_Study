import torch
import time
import torch.nn as nn
import torch.optim as optim
import sys
import os

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter

log_dir = '/media/user/HDD2/sh/Chung_Ang/Week1_Image_Classification/results/VGG/mnist'
base_filename = 'output_log'
file_index = 1

while os.path.exists(f"{log_dir}/{base_filename}_{file_index}.txt"):
    file_index += 1

sys.stdout = open(f"{log_dir}/{base_filename}_{file_index}.txt", 'w')

# Custom Dataset class for MNIST
class CustomMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = MNIST(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load training and test datasets
train_dataset = CustomMNISTDataset(root='/media/user/HDD2/sh/Chung_Ang/Week1_Image_Classification/data', train=True, transform=transform)
test_dataset = CustomMNISTDataset(root='/media/user/HDD2/sh/Chung_Ang/Week1_Image_Classification/data', train=False, transform=transform)

# DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the VGG16 model and modify the classifier for MNIST (10 classes)
model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features=num_ftrs, out_features=10)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Set up TensorBoard
writer = SummaryWriter(logdir='/media/user/SH/Chung_Ang/Week1_Image_Classification/runs/mnist_experiment')

# Training function
def train(model, train_loader, criterion, optimizer):

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

    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test(model, test_loader, criterion)

    writer.add_scalars('VGG_MNIST_Loss', {'Train Loss': train_loss, 'Test Loss': test_loss}, epoch + 1)
    writer.add_scalars('VGG_MNIST_Accuracy', {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}, epoch + 1)

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

hours, remainder = divmod(total_elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Close the TensorBoard writer
writer.close()

sys.stdout.close()
