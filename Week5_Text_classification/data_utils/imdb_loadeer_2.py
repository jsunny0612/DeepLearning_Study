import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer
from datasets import load_dataset


class IMDBDataset(Dataset):
    def __init__(self, max_length=250, split_type='train'):

        dataset = load_dataset("imdb")

        if split_type in ['train', 'val']:
            data = dataset['train']
            texts = data['text']
            labels = data['label']

            val_size = int(0.1 * len(texts))
            train_size = len(texts) - val_size
            train_indices, val_indices = random_split(range(len(texts)), [train_size, val_size])

            if split_type == 'train':
                self.dataset = [(texts[i], labels[i]) for i in train_indices]
            elif split_type == 'val':
                self.dataset = [(texts[i], labels[i]) for i in val_indices]

        elif split_type == 'test':
            data = dataset['test']
            texts = data['text']
            labels = data['label']
            self.dataset = list(zip(texts, labels))

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, label = self.dataset[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding.get('token_type_ids', None)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': torch.tensor(label, dtype=torch.long)
        }