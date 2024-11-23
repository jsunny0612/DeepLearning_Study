import pandas as pd
import torch
from torch.utils.data import Dataset
import re

class IMDBDataset(Dataset):
    def __init__(self, file_path, max_len=250, vocab =None):
        self.data = pd.read_csv(file_path)
        self.texts = []  # 전처리된 텍스트를 저장할 리스트
        self.vocab= {}  # 어휘 사전
        self.word_index = 1

        # 전처리 후 texts에 저장
        for text in self.data['text']:
            processed_text = self.preprocess_text(text)
            self.texts.append(processed_text)
            self.build_vocab(processed_text)

        self.labels = self.data['label'].values
        self.max_len = max_len

    def preprocess_text(self, text):
        text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
        text = text.replace('""', '')  # 이중 따옴표 제거
        text = re.sub(r'[^\w\s]', '', text).lower()  # 특수문자 제거 및 소문자 변환
        return text

    def build_vocab(self, text):
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.word_index
                self.word_index += 1

    def text_to_sequence(self, text):
        sequence = []
        vocab = {}
        word_index = 1
        for word in text.split():
            if word not in vocab:
                vocab[word] = word_index
                word_index += 1
            sequence.append(vocab[word])
        return sequence

    def pad_sequence(self, sequence):
        if len(sequence) < self.max_len:
            padding_size = self.max_len -len(sequence)
            sequence.extend([0]*padding_size)
        else:
            sequence = sequence[:self.max_len]
        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        raw_text = self.texts[idx]
        label = self.labels[idx]

        token_ids = self.text_to_sequence(raw_text)
        padded_token_ids = self.pad_sequence(token_ids)

        padded_token_ids_tensor = torch.tensor(padded_token_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {"input_ids": padded_token_ids_tensor, "labels": label_tensor}
