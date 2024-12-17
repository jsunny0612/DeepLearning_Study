import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from data_utils.wmt16_loader import WMT16Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer for Translation")
    parser.add_argument('--output', type=str, required=True, help="Path to save log file")
    return parser.parse_args()


def set_log(output):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    log_path = os.path.join(output, f"output_log_{current_time}.txt")
    print(f"Log file path: {log_path}")
    return log_path


def set_dataloader(batch_size=32, max_len=100, model_name="t5-small"):
    train_dataset = WMT16Dataset(split="train", max_length=max_len, model_name=model_name)
    val_dataset = WMT16Dataset(split="validation", max_length=max_len, model_name=model_name)
    test_dataset = WMT16Dataset(split="test", max_length=max_len, model_name=model_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.tokenizer


def set_model(model_name, device):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_config(config)
    model.to(device)
    return model

def train(model, train_loader, optimizer, device, teacher_forcing=True):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader):
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        if teacher_forcing:
            outputs = model(input_ids=source, labels=target)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, val_loader, device, teacher_forcing=True):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            if teacher_forcing:
                outputs = model(input_ids=source, labels=target)

            loss = outputs.loss
            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def test(model, test_loader, tokenizer, device, max_len, data_results, teacher_forcing=True):
    model.eval()
    test_loss = 0
    total_bleu = 0
    smoothing_function = SmoothingFunction().method1
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            if teacher_forcing:
                outputs = model(input_ids=source, labels=target)

            loss = outputs.loss
            test_loss += loss.item()

            generated_tokens = model.generate(source, max_length=max_len)
            for i in range(target.size(0)):
                ref = tokenizer.decode(target[i], skip_special_tokens=True).strip().split()
                hyp = tokenizer.decode(generated_tokens[i], skip_special_tokens=True).strip().split()

                bleu4 = sentence_bleu([ref], hyp, weights=(0.5, 0.25, 0.25, 0.0), smoothing_function=smoothing_function)
                total_bleu += bleu4
                num_samples += 1

                if len(data_results["test"]) < 100:  # Save up to 100 results for analysis
                    data_results["test"].append({"ref": " ".join(ref), "hyp": " ".join(hyp)})

    avg_loss = test_loss / len(test_loader)
    avg_bleu = total_bleu / num_samples
    print(f"Test Loss: {avg_loss:.4f}, BLEU-4: {avg_bleu:.2f}")
    return avg_loss, avg_bleu


def main():
    args = parse_args()

    log_file_path = set_log(args.output)
    sys.stdout = open(log_file_path, 'w')
    data_results = {"test": []}

    writer = SummaryWriter(logdir='./runs/wmt16_dataset_2')
    print("Modified Dataset: WMT16Dataset (Second)")
    print("This is a Transformer model from hugging-face.")

    train_loader, val_loader, test_loader, tokenizer = set_dataloader(
        batch_size=32, max_len=100, model_name="t5-small"
    )

    device = torch.device('cuda')
    model = set_model(model_name="t5-small", device=device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    epochs = 10
    max_len = 100
    total_start_time = time.time()
    for epoch in range(epochs):
        print(f"\n======= Epoch {epoch + 1}/{epochs} =======")
        train_loss = train(model, train_loader, optimizer, device, teacher_forcing=True)
        val_loss = validate(model, val_loader, device, teacher_forcing=True)

        writer.add_scalars("Hugging_Transformer_Loss_2", {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
        scheduler.step()

    print("\n======= Final Test Evaluation =======")
    test_loss, test_bleu = test(model, test_loader, tokenizer, device, max_len, data_results, teacher_forcing=True)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test BLEU-4 Score: {test_bleu:.2f}%")

    output_dir = "./data_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_filepath = os.path.join(output_dir, "hugging_transformer_test_results_2.json")
    with open(results_filepath, "w") as f:
        json.dump(data_results["test"], f, indent=4)
    print(f"Final Test Results saved to {results_filepath}")

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    writer.close()
    sys.stdout.close()


if __name__ == '__main__':
    main()
