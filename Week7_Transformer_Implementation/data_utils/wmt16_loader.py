from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class WMT16Dataset(Dataset):
    def __init__(self, split, max_length, model_name="t5-small", lang_pair=("en", "de")):

        self.data = self.load_data(split, lang_pair)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def load_data(self, split, lang_pair):

        dataset = load_dataset("bentrevett/multi30k", split=split)
        source_lang, target_lang = lang_pair

        data_pairs = []
        for example in dataset:
            source_text = example[source_lang]
            target_text = example[target_lang]
            data_pairs.append((source_text, target_text))

        return data_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        source_text, target_text = self.data[idx]

        source = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "source": source["input_ids"].squeeze(0),
            "target": target["input_ids"].squeeze(0)
        }
