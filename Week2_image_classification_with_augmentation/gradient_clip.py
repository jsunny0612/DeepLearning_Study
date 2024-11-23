import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split

from data.MNIST.mnist import CustomMNIST
from data.cifar10.cifar10 import CustomCIFAR10
from model.ResNet.resnet18 import resnet18
from model.VGG.vgg16 import vgg16


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test")
    parser.add_argument('--model', type=str, required=True, help="choose model")
    parser.add_argument('--dataset', type=str, required=True, help="choose dataset")
    parser.add_argument('--data_path', type=str, required=True, help="path to data file")
    parser.add_argument('--output', type=str, required=True, help="path to output file")
    return parser.parse_args()


def set_log(output):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(output):
        os.makedirs(output)
    log_file_name = f"output_log_{current_time}.txt"
    log_file_path = os.path.join(output, log_file_name)
    return log_file_path


def set_dataloader(dataset, data_path):
    if dataset == 'mnist':
        train_dataset = CustomMNIST(root=data_path, train=True)
        test_dataset = CustomMNIST(root=data_path, train=False)
    elif dataset == 'cifar10':
        train_dataset = CustomCIFAR10(root=data_path, train=True)
        test_dataset = CustomCIFAR10(root=data_path, train=False)

    dataset_size = len(train_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def set_model(model_name):
    if model_name == 'vgg16':
        return vgg16(num_classes=10)
    elif model_name == 'resnet18':
        return resnet18(num_classes=10)


def train(model, train_loader, criterion, optimizer, device, max_grad_norm=None):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # 손실 및 정확도 계산
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100. * correct_train / total_train
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct_val / total_val
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    return val_loss, val_accuracy


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct_test / total_test
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy


def main():
    args = parse_args()
    log_file_path = set_log(args.output)
    sys.stdout = open(log_file_path, 'w')

    writer = SummaryWriter(logdir='/media/user/SH/Chung_Ang/Week2_image_classification_with_augmentation/runs/argparser_experiment')

    train_loader, val_loader, test_loader = set_dataloader(args.dataset, args.data_path)
    model = set_model(args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 10
    total_start_time = time.time()

    for epoch in range(epochs):
        print(f"\n======= Epoch {epoch + 1}/{epochs} =======")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, max_grad_norm=5.0)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        loss_name = f'Grad_Clip_{args.dataset}_{args.model}_Loss'
        accuracy_name = f'Grad_Clip_{args.dataset}_{args.model}_Accuracy'

        writer.add_scalars(loss_name, {'Train': train_loss, 'Validation': val_loss, 'Test': test_loss}, epoch + 1)
        writer.add_scalars(accuracy_name, {'Train': train_acc, 'Validation': val_acc, 'Test': test_acc}, epoch + 1)

        # Scheduler step
        scheduler.step()

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    writer.close()
    sys.stdout.close()


if __name__ == '__main__':
    main()
