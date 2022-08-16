import torch
import pickle
import random
import numpy as np
import sys
from IPython import embed
import os

from tqdm import tqdm
import argparse

import csv
import sys
sys.path.append("..")

from util import negative_sampling, compute_f1
from dataset import Dataset, DatasetwithGraph
from bertbasic import BertBasic

from pytorch_pretrained_bert import BertAdam
from sklearn.model_selection import train_test_split, KFold

from transformers import BertTokenizer, BertForMaskedLM


parser = argparse.ArgumentParser(description='Train the Bert_with_graph model.')
parser.add_argument('--lr', type=float, default=1e-4,
                         help='an integer for the accumulator')
parser.add_argument('--wd', type=float, default=1e-4,
                         help='weight decay')
parser.add_argument('--data', type=int, default=0,
                         help='dataset choice')
parser.add_argument('--neg_sam', type=float, default=0.4,
                         help='negtive sampling rate')                        

# parser.add_argument('lr', type=float, default=1e-3,
#                          help='an integer for the accumulator') 

seed = 1236
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def load_embeddings(pretrain_embed_path):
    print("Loading glove concept embeddings with pooling:", pretrain_embed_path)
    concept_vec = np.load(pretrain_embed_path)
    print("done!")
    return concept_vec

def negative_sampling(data, pos= 5, ratio=0.7):
    result = []
    for d in data:
        if d[pos] == 'NULL':
            if random.random() < ratio:
                continue
        result.append(d)
    return result

if __name__ == '__main__':
    device = torch.device("cuda")
    args = parser.parse_args()

    v_prf_sum = []
    t_prf_sum = []
    t_prf_epoch = []
    v_prf_epoch = []

    if args.data == 0:
        data_file = 'causaltb.pickle' # onlybert preprocessed data: data_onlybert_0.9.pickle'                                 # '/u/wusifan/Event_causality_identification/cedar/data_concept_graph_e_v0.9.pickle' this
        ### Reading event story line data...
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    
    train_set, test_set = train_test_split(data, test_size=0.1)
    train_set = negative_sampling(train_set)

    ###
    test_dataset = Dataset(16, test_set)
    test_dataset_batch = list([batch for batch in test_dataset.reader(device, True)])

    ###
    train_dataset = Dataset(16, train_set)
    train_dataset_batch = list([batch for batch in train_dataset.reader(device, True)])
    checkpoint_path = "checkpoint/bert_test/"

    model = BertBasic(y_num=3, hiddensize=768).to(device)
    
    learning_rate = 1e-5
    optimizer = BertAdam(model.parameters(), lr=learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    m = torch.nn.Sigmoid()
    loss_bce = torch.nn.BCELoss()

    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_precision_v = 0
    best_recall_v = 0
    best_f1_v = 0
    stop = 0
    lossMIN = 10000000
    for epoch in range(0, 10):
        idx = 0
        predicted_all = []
        gold_all = []
        for batch in tqdm(train_dataset_batch, mininterval=2, total=len(train_dataset_batch), file=sys.stdout, ncols=80):
            idx += 1
            model.train()

            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
            
            logits, cls1, cls2, new_concept_embed = model.forward(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
           
            loss = loss_fn(logits, data_y) 
            predicted = torch.argmax(logits, -1)
            predicted = list(predicted.cpu().numpy())
            predicted_all += predicted

            gold = list(data_y.cpu().numpy())
            gold_all += gold
        

            optimizer.zero_grad()
            #optimizer_mask.zero_grad()
            loss.backward()
            optimizer.step()
            #optimizer_mask.step()
        print("experimenting test_bert model on causaltb")
        print("epoch ", epoch, "-th training results: ")
        p, r, f = compute_f1(gold_all, predicted_all)
        t_prf_epoch.append([p, r, f])
        print("training precision: %f, recall: %f, f1: %f" %(p, r, f))

        model.eval()

        with torch.no_grad():
            predicted_all = []
            gold_all = []
            for batch in test_dataset_batch:
                sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch


                logits, cls1, cls2, new_concept_embed = model.forward(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
                

                predicted = torch.argmax(logits, -1)
                predicted = list(predicted.cpu().numpy())
                predicted_all += predicted

                gold = list(data_y.cpu().numpy())
                gold_all += gold

            print("testing results:")
            p, r, f = compute_f1(gold_all, predicted_all)
            v_prf_epoch.append([p, r, f])
            if f > best_f1: 
                best_f1 = f 
                best_precision = p 
                best_recall = r
                save_path = checkpoint_path + str("%.4f" % best_f1) + '.pth.tar'
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict()},
                           save_path)

            print("testing precision: %f, recall: %f, f1: %f" %(p, r, f))
            print("best testing precision: %f, best recall: %f, best f1: %f" %(best_precision,best_recall, best_f1))


    print("bert basic model")
    print("dataset from ", data_file)
