import torch
import time
import torch.nn as nn
import torch.optim as optim
import sys
import os

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tensorboardX import SummaryWriter
from PIL import Image

log_dir = '/media/user/SH/Chung_Ang/Week1_Image_Classification/results/VGG/cifar10'
base_filename = 'output_log'
file_index = 1

while os.path.exists(f"{log_dir}/{base_filename}_{file_index}.txt"):
    file_index += 1

sys.stdout = open(f"{log_dir}/{base_filename}_{file_index}.txt", 'w')

# Custom Dataset class for CIFAR-10
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data_path = os.path.join(root, 'train' if train else 'test')
        self.transform = transform
        self.images = []
        self.labels = []

        # Load images and labels
        for class_idx, class_name in enumerate(sorted(os.listdir(self.data_path))):
            class_path = os.path.join(self.data_path, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load training and test datasets
train_dataset = CustomCIFAR10Dataset(root='/media/user/SH/Chung_Ang/Week1_Image_Classification/data/cifar10/', train=True, transform=transform)
test_dataset = CustomCIFAR10Dataset(root='/media/user/SH/Chung_Ang/Week1_Image_Classification/data/cifar10/', train=False, transform=transform)

# DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the VGG16 model and modify the classifier for CIFAR-10 (10 classes)
model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features=num_ftrs, out_features=10)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Set up TensorBoard
writer = SummaryWriter(logdir='/media/user/SH/Chung_Ang/Week1_Image_Classification/runs/cifar10_experiment')

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

    # Log to TensorBoard
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
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy

# Main training and testing loop
epochs = 10
total_start_time = time.time()

for epoch in range(epochs):
    print(f"\n======= Data_Aug_VGG_CIFAR10_Epoch {epoch + 1}/{epochs} =======")
    print(f'Epoch {epoch + 1}:')

    train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = test(model, test_loader, criterion, epoch)

    writer.add_scalars('Data_Aug_VGG_CIFAR10_Loss', {'Train Loss': train_loss, 'Test Loss': test_loss}, epoch + 1)
    writer.add_scalars('Data_Aug_VGG_CIFAR10_Accuracy', {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}, epoch + 1)

total_end_time = time.time()
total_elapsed_time = total_end_time - total_start_time

hours, remainder = divmod(total_elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Close the TensorBoard writer
writer.close()
sys.stdout.close()
