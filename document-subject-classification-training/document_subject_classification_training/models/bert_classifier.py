from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, model_name, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        #the output label number has be be dynamic
        self.linear = nn.Linear(768, 56)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def create(model_name):

    return BertClassifier(model_name)
