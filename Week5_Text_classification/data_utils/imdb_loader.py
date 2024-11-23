'''
import pandas as pd
import torch
from torch.utils.data import Dataset
import re
from transformers import AutoTokenizer


class IMDBDataset(Dataset):
    def __init__(self, file_path, model_type='bert', max_len=300, vocab=None):
        self.data = pd.read_csv(file_path)
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].values
        self.max_len = max_len
        self.model_type = model_type

        if model_type == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        elif model_type in ['lstm', 'transformer_encoder']:
            self.vocab = {} if vocab is None else vocab
            self.word_index = 1
            if not vocab:
                for text in self.data['text']:
                    processed_text = self.preprocess_text(text)
                    self.build_vocab(processed_text)
            self.vocab_size = len(self.vocab)

    def preprocess_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = text.replace('""', '')
        text = re.sub(r'[^\w\s]', '', text).lower()
        return text

    def build_vocab(self, text):
        word_freq = {}
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)[:10000]
        for word in sorted_words:
            if word not in self.vocab:
                self.vocab[word] = self.word_index
                self.word_index += 1

    def text_to_sequence(self, text):
        return [self.vocab.get(word, 0) for word in text.split()]

    def pad_sequence(self, sequence):
        if len(sequence) < self.max_len:
            sequence.extend([0] * (self.max_len - len(sequence)))
        else:
            sequence = sequence[:self.max_len]
        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.model_type == 'bert':
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }

        elif self.model_type in ['transformer_encoder']:
            token_ids = self.text_to_sequence(text)
            padded_token_ids = self.pad_sequence(token_ids)
            attention_mask = [1 if token_id != 0 else 0 for token_id in padded_token_ids]
            return {
                "input_ids": torch.tensor(padded_token_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long)
            }

        elif self.model_type == 'lstm':
            token_ids = self.text_to_sequence(text)
            padded_token_ids = self.pad_sequence(token_ids)
            return {
                "input_ids": torch.tensor(padded_token_ids, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long)
            }
'''

import pandas as pd
import torch
from torch.utils.data import Dataset
import re
from transformers import AutoTokenizer


class IMDBDataset(Dataset):
    def __init__(self, file_path, model_type='bert', max_len=300, vocab=None):
        self.data = pd.read_csv(file_path)
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].values
        self.max_len = max_len
        self.model_type = model_type

        if model_type == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        elif model_type in ['lstm', 'transformer_encoder']:
            self.vocab = {} if vocab is None else vocab
            self.word_index = 1
            if not vocab:
                for text in self.data['text']:
                    processed_text = self.preprocess_text(text)
                    self.build_vocab(processed_text)
            self._vocab_size = len(self.vocab)  # _vocab_size로 설정

    @property
    def vocab_size(self):
        return self._vocab_size

    def preprocess_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = text.replace('""', '')
        text = re.sub(r'[^\w\s]', '', text).lower()
        return text

    def build_vocab(self, text):
        word_freq = {}
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)[:10000]
        for word in sorted_words:
            if word not in self.vocab:
                self.vocab[word] = self.word_index
                self.word_index += 1

    def text_to_sequence(self, text):
        return [self.vocab.get(word, 0) for word in text.split()]

    def pad_sequence(self, sequence):
        if len(sequence) < self.max_len:
            sequence.extend([0] * (self.max_len - len(sequence)))
        else:
            sequence = sequence[:self.max_len]
        return sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.model_type == 'bert':
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }

        elif self.model_type in ['transformer_encoder']:
            token_ids = self.text_to_sequence(text)
            padded_token_ids = self.pad_sequence(token_ids)
            attention_mask = [1 if token_id != 0 else 0 for token_id in padded_token_ids]
            return {
                "input_ids": torch.tensor(padded_token_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long)
            }

        elif self.model_type == 'lstm':
            token_ids = self.text_to_sequence(text)
            padded_token_ids = self.pad_sequence(token_ids)
            return {
                "input_ids": torch.tensor(padded_token_ids, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.long)
            }

