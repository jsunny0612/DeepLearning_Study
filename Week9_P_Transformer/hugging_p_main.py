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

def set_model(model_name, num_encoders, num_decoders):
    # 기본 T5 모델 로드
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    base_encoder = base_model.get_encoder()
    base_decoder = base_model.get_decoder()

    # 병렬 Encoders와 Decoders 생성
    encoders = nn.ModuleList([base_encoder for _ in range(num_encoders)])
    decoders = nn.ModuleList([base_decoder for _ in range(num_decoders)])

    # Hidden size와 Vocabulary size
    hidden_size = base_encoder.config.hidden_size
    vocab_size = base_decoder.config.vocab_size

    # 결합 레이어 정의
    combination_layer = nn.Linear(num_decoders * hidden_size, hidden_size)

    class PTransformer(nn.Module):
        def __init__(self):
            super(PTransformer, self).__init__()
            self.encoders = encoders
            self.decoders = decoders
            self.combination_layer = combination_layer

        def forward(self, source, target):
            # 병렬 Encoder 출력 계산
            encoder_outputs = [
                encoder(input_ids=source, return_dict=True).last_hidden_state
                for encoder in self.encoders
            ]

            # 병렬 Decoder 출력 계산
            decoder_outputs = [
                decoder(
                    input_ids=target[:, :-1], encoder_hidden_states=encoder_outputs[i], return_dict=True
                ).last_hidden_state
                for i, decoder in enumerate(self.decoders)
            ]

            # Decoder 출력 결합
            combined_output = torch.cat(decoder_outputs, dim=-1)  # Concatenate outputs
            combined_output = self.combination_layer(combined_output)  # Linear layer to reduce dimensions

            return combined_output

    return PTransformer()

def train(model, train_loader, criterion, optimizer, device, teacher_forcing=True):
    model.train()
    train_loss = 0

    for batch in tqdm(train_loader):
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        if teacher_forcing:
            outputs = model(source, target)

        loss = criterion(outputs.view(-1, outputs.size(-1)), target[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            input_token = target[:, 0].unsqueeze(1)
            outputs = []

            for t in range(1, target.size(1)):
                output = model(input_ids=source, decoder_input_ids=input_token, return_dict=True).logits
                outputs.append(output[:, -1, :])
                input_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)

            outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), target[:, 1:].contiguous().view(-1))
            val_loss += loss.item()

    avg_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def test(model, test_loader, tokenizer, device, max_len, data_results):
    model.eval()
    test_loss = 0
    total_bleu = 0
    smoothing_function = SmoothingFunction().method1
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            input_token = target[:, 0].unsqueeze(1)  # 시작 토큰
            outputs = []

            for t in range(1, target.size(1)):
                output = model(input_ids=source, decoder_input_ids=input_token, return_dict=True).logits
                outputs.append(output[:, -1, :])
                input_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)

            outputs = torch.stack(outputs, dim=1)
            generated_tokens = outputs.argmax(dim=-1)
            for i in range(target.size(0)):
                ref = tokenizer.decode(target[i].tolist(), skip_special_tokens=True).split()
                hyp = tokenizer.decode(generated_tokens[i].tolist(), skip_special_tokens=True).split()

                bleu4 = sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
                total_bleu += bleu4
                num_samples += 1

    avg_bleu = total_bleu / num_samples
    print(f"Test BLEU-4 Score: {avg_bleu:.2f}%")
    return avg_bleu


def main():
    args = parse_args()

    log_file_path = set_log(args.output)
    sys.stdout = open(log_file_path, 'w')
    data_results = {"test": []}

    writer = SummaryWriter(logdir='./runs/wmt16_dataset')
    print("Modified Dataset: WMT16Dataset")
    print("This is a P-Transformer from loading model.")

    train_loader, val_loader, test_loader, tokenizer = set_dataloader(
        batch_size=32, max_len=100, model_name="t5-small")

    device = torch.device('cuda')
    model = set_model(model_name="t5-small", num_encoders=3, num_decoders=3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    epochs = 10
    max_len = 100
    total_start_time = time.time()
    for epoch in range(epochs):
        print(f"\n======= Epoch {epoch + 1}/{epochs} =======")
        train_loss = train(model, train_loader, criterion, optimizer, device, teacher_forcing=True)
        val_loss = validate(model, val_loader, device)

        writer.add_scalars("Loading_P_Transformer_Loss", {'Train': train_loss, 'Validation': val_loss}, epoch + 1)
        scheduler.step()

    print("\n======= Final Test Evaluation =======")
    test_loss, test_bleu = test(model, test_loader, tokenizer, device, max_len, data_results)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test BLEU-4 Score: {test_bleu:.2f}%")

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    writer.close()
    sys.stdout.close()


if __name__ == '__main__':
    main()
