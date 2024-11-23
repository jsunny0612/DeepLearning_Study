import argparse
import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from data_utils.imdb_loadeer_2 import IMDBDataset
from classification_model.bert_model import PretrainedBERTModel
from classification_model.lstm_model import LSTM
from classification_model.transformer_encoder_model import *


def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification")
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer_encoder', 'bert'], required=True,
                        help="Choose model")
    parser.add_argument('--output', type=str, required=True, help="Path to output file")
    return parser.parse_args()


def set_log(output):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(output):
        os.makedirs(output)
    return os.path.join(output, f"output_log_{current_time}.txt")


def set_dataloader(batch_size=64, max_len=250):
    train_dataset = IMDBDataset(split_type='train', max_length=max_len)
    val_dataset = IMDBDataset(split_type='val', max_length=max_len)
    test_dataset = IMDBDataset(split_type='test', max_length=max_len)

    vocab_size = train_dataset.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab_size


def set_model(model_type='lstm', vocab_size=None):
    if model_type == 'lstm':
        return LSTM(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_size=512,
            num_layers=2,
            dropout=0.5,
            bidirectional=True,
            num_classes=2,
        )
    elif model_type == 'transformer_encoder':
        return TransformerEncoder(vocab_size=vocab_size, embedding_dim=128, num_classes=2)
    elif model_type == 'bert':
        return PretrainedBERTModel()


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)

        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = 100. * correct_train / total_train
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct_val / total_val
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    return val_loss, val_accuracy


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss, correct_test, total_test = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct_test / total_test
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy


def main():
    args = parse_args()

    log_file_path = set_log(args.output)
    sys.stdout = open(log_file_path, 'w')

    writer = SummaryWriter(logdir='/media/user/HDD2/sh/Chung_Ang/Week5_Text_classification/runs/imdb_experiment')
    print(f"Chosen Model: {args.model}")

    train_loader, val_loader, test_loader, vocab_size = set_dataloader(batch_size=64, max_len=250)
    model = set_model(model_type=args.model, vocab_size=vocab_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    epochs = 10
    total_start_time = time.time()
    for epoch in range(epochs):
        print(f"\n======= Epoch {epoch + 1}/{epochs} =======")

        # Train and validate each epoch
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()

        # Log scalars to TensorBoard for each epoch
        writer.add_scalars(f'{args.model}_Loss', {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
        writer.add_scalars(f'{args.model}_Accuracy', {'Train': train_acc, 'Validation': val_acc}, epoch + 1)

    # Final test evaluation after all epochs
    print("\n======= Final Test Evaluation =======")
    test_loss, test_acc = test(model, test_loader, criterion, device)

    writer.add_scalar(f'{args.model}_Final_Test_Loss', test_loss)
    writer.add_scalar(f'{args.model}_Final_Test_Accuracy', test_acc)

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    writer.close()
    sys.stdout.close()


if __name__ == '__main__':
    main()
