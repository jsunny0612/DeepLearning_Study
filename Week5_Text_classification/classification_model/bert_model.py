import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class PretrainedBERTModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super(PretrainedBERTModel, self).__init__()

        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.fc = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        return output.logits

