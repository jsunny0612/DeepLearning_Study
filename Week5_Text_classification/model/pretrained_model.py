# model/pretrained_model.py
'''
import torch
import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', output_size=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)
'''

import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', output_size=2, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        # Get last_hidden_state and pooler_output from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use the [CLS] token embedding for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Apply dropout and pass through the fully connected layer
        cls_output = self.dropout(cls_output)
        return self.fc(cls_output)
