import numpy as np
import pickle
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertAdam

from dataset import Dataset

class BertBasic(nn.Module):
    def __init__(self, y_num, hiddensize):
        super(BertBasic, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(768 * 2, y_num)

        self.additional_fc = nn.Linear(768 * 4, y_num)


    def forward(self, sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask):

        if self.training:
            self.bert.train()
            encoded_layers_s, _ = self.bert(sentences_s, attention_mask=mask_s)
            enc_s = encoded_layers_s[-1]

            #encoded_layers_t, _ = self.bert(sentences_t, attention_mask=mask_t)
            #enc_t = encoded_layers_t[-1]

        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers_s, _ = self.bert(sentences_s, attention_mask=mask_s)
                enc_s = encoded_layers_s[-1]

                #encoded_layers_t, _ = self.bert(sentences_t, attention_mask=mask_t)
                #enc_t = encoded_layers_t[-1]


        event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(enc_s, event1)])
        event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(enc_s, event2)])

        m1 = event1_mask.unsqueeze(-1).expand_as(event1).float()
        m2 = event2_mask.unsqueeze(-1).expand_as(event2).float()

        event1 = event1 * m1
        event2 = event2 * m2

        opt1 = torch.sum(event1, dim=1)
        opt2 = torch.sum(event2, dim=1)

        opt = torch.cat((opt1, opt2), 1)
        opt = self.drop(opt)
        opt = self.fc(opt)
        return opt, opt1, opt2, opt