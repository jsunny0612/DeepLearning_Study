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
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data_utils.wmt16_loader import WMT16Dataset
from model.P_transformer import P_Transformer


def parse_args():
    parser = argparse.ArgumentParser(description="P_Transformer for Translation")
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


def set_model(n_input_vocab, n_output_vocab, d_model, head, d_ff, max_len, padding_idx,
              dropout, n_layers, n_parallel_encoders, n_parallel_decoders, device):
    return P_Transformer(n_input_vocab=n_input_vocab,
                         n_output_vocab=n_output_vocab,
                         d_model=d_model,
                         head=head,
                         d_ff=d_ff,
                         max_len=max_len,
                         padding_idx=padding_idx,
                         dropout=dropout,
                         n_parallel_encoders=n_parallel_encoders,
                         n_parallel_decoders=n_parallel_decoders,
                         n_layers=n_layers,
                         device=device)


def greedy_search(outputs, tokenizer):
    generated_sequences = []
    batch_size, seq_len, vocab_size = outputs.size()

    for batch_idx in range(batch_size):
        sequence = []
        for step in range(seq_len):
            token_id = torch.argmax(outputs[batch_idx, step], dim=-1).item()
            if token_id == tokenizer.eos_token_id:
                break
            sequence.append(token_id)
        generated_sequences.append(sequence)
    return generated_sequences

def train(model, train_loader, criterion, optimizer, device, teacher_forcing=True):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader):
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        if teacher_forcing:
            outputs = model(source, target[:, :-1])

        loss = criterion(outputs.view(-1, outputs.size(-1)), target[:, 1:].contiguous().view(-1))
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f'Train Loss: {train_loss:.4f}')
    return train_loss


def validate(model, val_loader, tokenizer, criterion, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            input_token = target[:, :1]
            outputs = []

            batch_size, max_length = target.size()

            for t in range(1, max_length):
                output = model(source, input_token)  # (batch_size, seq_len, vocab_size)
                outputs.append(output[:, -1, :])
                input_token = output[:, -1, :].argmax(dim=-1, keepdim=True)  # GREEDY SEARCH

                if (input_token == tokenizer.eos_token_id).all():
                    break

            outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len-1, vocab_size)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target[:, 1:].contiguous().view(-1))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')
    return val_loss


def test(model, test_loader, criterion, tokenizer, device):
    model.eval()
    test_loss = 0
    total_bleu4 = 0
    num_bleu_samples = 0
    smoothing_function = SmoothingFunction().method1

    with torch.no_grad():
        for batch in tqdm(test_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            input_token = target[:, :1]
            outputs = []

            batch_size, max_length = target.size()

            for t in range(1, batch_size):
                output = model(source, input_token)
                outputs.append(output[:, -1, :])
                input_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

                if(input_token == tokenizer.eos_token_id).all():
                    break

            outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len-1, vocab_size)
            loss = criterion(outputs.view(-1, outputs.size(-1)), target[:, 1:].contiguous().view(-1))
            test_loss += loss.item()

            for i in range(target.size(0)):
                ref = tokenizer.decode(target[i].tolist(), skip_special_tokens=True).split()
                hyp = tokenizer.decode(outputs[i].argmax(dim=-1).tolist(), skip_special_tokens=True).split()

                if "<EOS>" in hyp:
                    hyp = hyp[:hyp.index("<EOS>")]

                bleu4 = sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25),
                                      smoothing_function=smoothing_function)
                total_bleu4 += bleu4
                num_bleu_samples += 1

    test_loss /= len(test_loader)
    avg_bleu_score4 = (total_bleu4 / num_bleu_samples)
    print(f"Test Loss: {test_loss:.4f}, Test BLEU-4 Score: {avg_bleu_score4:.2f}")
    return test_loss, avg_bleu_score4


def main():
    args = parse_args()
    log_file_path = set_log(args.output)
    sys.stdout = open(log_file_path, 'w')
    writer = SummaryWriter(logdir='./runs/P_Transformer')

    train_loader, val_loader, test_loader, tokenizer = set_dataloader(batch_size=32, max_len=100, model_name="t5-small")
    vocab_size = tokenizer.vocab_size

    device = torch.device('cuda')
    model = set_model(n_input_vocab=vocab_size, n_output_vocab=vocab_size,
                      d_model=512, head=8, d_ff=2048, max_len=100,
                      padding_idx=tokenizer.pad_token_id, dropout=0.3,
                      n_layers=6, n_parallel_encoders=6, n_parallel_decoders=6, device=device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

    epochs = 10
    total_start_time = time.time()
    for epoch in range(epochs):
        print(f"\n======= Epoch {epoch + 1}/{epochs} =======")
        train_loss = train(model, train_loader, criterion, optimizer, device, teacher_forcing=True)
        val_loss = validate(model, val_loader, tokenizer, criterion, device)
        writer.add_scalars("Metrics", {'Train Loss': train_loss, 'Validation Loss': val_loss}, epoch + 1)
        scheduler.step()

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    test_loss, test_bleu = test(model, test_loader, criterion, tokenizer, device)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test BLEU-4 Score: {test_bleu:.2f}")
    writer.close()
    sys.stdout.close()



if __name__ == '__main__':
    main()