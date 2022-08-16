import torch
import pickle
import random
import numpy as np
import os

from IPython import embed

from tqdm import tqdm
import argparse
import networkx as nx
import dgl

import csv
import sys
sys.path.append("..")

from util import negative_sampling, compute_f1, load_checkpoint
from dataset import Dataset, DatasetwithGraph
from bertbasic import BertBasic
from Graph import GraphEnhanceNetworks

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

def my_topk(a, b, k):
    ## N * F, F
    # Return: k
    sim = torch.nn.CosineSimilarity()
    n = a.shape[0]
    b = b.repeat(n).view(n,-1)
    c = sim(a, b) # N
    dis, indices = torch.topk(c, k) # 
    return dis, indices

def test(all_list, inputs, model, k, device):
    dgraph = []
    model.eval()
    sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = inputs
    with torch.no_grad():
        logits, cls1, cls2, new_concept_embed = model.forward(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
        bsz, dim = cls1.shape
        near_id = torch.zeros(k)

        res1 = torch.zeros(bsz, k+1, dim)
        res1[:, 0, :] = cls1
        res2 = torch.zeros(bsz, k+1, dim)
        res2[:, 0, :] = cls2
        for i in range(bsz):
            graph = nx.DiGraph()

            key1 = cls1[i].cpu()
            key2 = cls2[i].cpu()

            dis1, near_id1 = my_topk(all_list, key1, k)
            res1[i,1:,:] = torch.index_select(all_list, 0, near_id1)
            
            dis2, near_id2 = my_topk(all_list, key2, k)
            res2[i,1:,:] = torch.index_select(all_list, 0, near_id2)
            
            for j in range(k):
                graph.add_edge(key1, res1[i,j+1,:], weight=dis1[j])
                graph.add_edge(key2, res2[i,j+1,:], weight=dis2[j])
    
            nxg = nx.convert_node_labels_to_integers(graph, label_attribute='feat')
            feas = [nxg.nodes[n_id]['feat'] for n_id in range(len(nxg))]
                     
            dg = dgl.from_networkx(nxg, node_attrs=['feat'], edge_attrs=['weight'])

            dgraph.append(dg)
    
    batch_graph = dgl.batch(dgraph)
    
    res1 = torch.mean(res1, 1).to(device)
    res2 = torch.mean(res2, 1).to(device)

    return res1, res2, batch_graph
       
def findk(all_list_s, all_list_k, index):
    bsz, k = index.shape
    out_s = torch.zeros(bsz, k, all_list_s.shape[-1])
    out_k = torch.zeros(bsz, k, all_list_k.shape[-1])
    for i in range(bsz):
        out_s[i] = torch.index_select(all_list_s, 0, index[i])
        out_k[i] = torch.index_select(all_list_k, 0, index[i])

    return out_s, out_k

def combinek(input, neark):
    out = np.expand_dims(input, 1)
    out = np.mean(np.concatenate((out, neark),1),1)
    return out

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

    pretrain_cpt_emd_path = "/u/wusifan/Event_causality_identification/cedar/embeddings/glove.transe.sgd.ent.npy"
    pretrain_rel_emd_path = "/u/wusifan/Event_causality_identification/cedar/embeddings/glove.transe.sgd.rel.npy"

    pretrained_concept_emd = load_embeddings(pretrain_cpt_emd_path)
    pretrained_relation_emd = load_embeddings(pretrain_rel_emd_path)
    print("pretrained_concept_emd.shape:", pretrained_concept_emd.shape)
    print("pretrained_relation_emd.shape:", pretrained_relation_emd.shape)

    # add one concept vec for dummy concept
    concept_dim = pretrained_concept_emd.shape[1]
    concept_num = pretrained_concept_emd.shape[0] + 1  # for dummy concept
    pretrained_concept_emd = np.insert(pretrained_concept_emd, 0, np.zeros((1, concept_dim)), 0)

    relation_num = pretrained_relation_emd.shape[0] * 2 + 1  # for inverse and dummy relations
    relation_dim = pretrained_relation_emd.shape[1]
    pretrained_relation_emd = np.concatenate((pretrained_relation_emd, pretrained_relation_emd))
    pretrained_relation_emd = np.insert(pretrained_relation_emd, 0, np.zeros((1, relation_dim)), 0)

    pretrained_concept_emd = torch.FloatTensor(pretrained_concept_emd)
    pretrained_relation_emd = torch.FloatTensor(pretrained_relation_emd)  # torch.FloatTensor(pretrained_relation_emd)

    k = 3
    MAXIMUM_SIZE = 10000
    E_DIMENSION =768
    S_DIMENSION = 768
    K_DIMENSION = 768
    G_DIMENSION = 768
    TOKEN_DIM = 105

    all_e = torch.zeros(MAXIMUM_SIZE, E_DIMENSION)
    all_s = torch.zeros(MAXIMUM_SIZE, S_DIMENSION)
    all_k = torch.zeros(MAXIMUM_SIZE, K_DIMENSION)
    all_g = torch.zeros(MAXIMUM_SIZE, G_DIMENSION)

    data_file = 'causaltb.pickle' 
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    train_set, test_set = train_test_split(data, test_size=0.1)
    train_set = negative_sampling(train_set)

    ###
    test_bsz = 10
    test_dataset = Dataset(test_bsz, test_set)
    test_batch = list([batch for batch in test_dataset.reader(device, False)])

    train_bsz = 20
    train_dataset = Dataset(train_bsz, train_set)
    train_batch = list([batch for batch in train_dataset.reader(device, False)])
   

    model_r = BertBasic(y_num=3, hiddensize=768).to(device)

    model_g = GraphEnhanceNetworks(y_num=3, hiddensize=768, 
                concept_dim=768, graph_hidden_dim=512, graph_output_dim=768, 
                pretrained_concept_emd=pretrained_concept_emd, concept_emd=768, device=device).to(device)

    best_f1 = 0.2195
    checkpoint_path = "checkpoint/test_bertknow2/" # train_bert_graph1: graph不参与retrieval过程, graph_e
    checkpoint_PATH = checkpoint_path + str("%.4f" % best_f1) + '.pth.tar'
    
    
    learning_rate = 1e-5
    optimizer = BertAdam(model_r.parameters(), lr=learning_rate)
    model, optimizer = load_checkpoint(model_r, checkpoint_PATH, optimizer)

    opt_fc = BertAdam(model.fc.parameters(), lr=learning_rate)
    opt_g = BertAdam(model_g.parameters(), lr=learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    v_prf_sum = []
    t_prf_sum = []
    test_prf_epoch = []
    train_prf_epoch = []


    finished_keys = 0
    for batch in tqdm(train_batch):
        model.eval()
        sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch

        with torch.no_grad():
            
            logits, cls1, cls2, new_concept_embed = model.forward(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
            cls_sum = torch.cat((cls1, cls2), 0)
            keys = cls_sum # [20, 768]
            
            bsz, dim = keys.shape

            all_k[finished_keys:finished_keys+bsz] = keys.cpu()
            finished_keys += bsz


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
    

    for epoch in range(40):
        predicted_all = []
        gold_all = []
        loss = 0
        model_g.train()
        for idx, sample in enumerate(tqdm(list(train_batch))):
            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = sample
            
            cls1, cls2, graph = test(all_k, sample, model, k, device)
            
            new_con_emb = model_g.forward(cls1.shape[0], graph)
            cls_mix = torch.cat((cls1, cls2, new_con_emb), dim=1)

            logits = model_g.classification(cls_mix.type(torch.cuda.FloatTensor))
            loss_step = loss_fn(logits, data_y)
            loss  += loss_step
            
            predicted = torch.argmax(logits, -1)
            predicted = list(predicted.cpu().numpy())
            predicted_all += predicted

            gold = list(data_y.cpu().numpy())
            gold_all += gold
            
            p_step, r_step, f_step = compute_f1(gold, predicted)
            v_prf_sum.append([p_step, r_step, f_step])

            #optimizer.zero_grad()
            opt_fc.zero_grad()
            opt_g.zero_grad()
            loss_step.backward()
            #optimizer.step()
            opt_fc.step()
            opt_g.step()

        p_t, r_t, f_t = compute_f1(gold_all, predicted_all)
        train_prf_epoch.append([p_t, r_t, f_t])
        print("train test_graoh ", epoch)
        print("training loss is ", loss)
        print("training precision: %f, recall: %f, f1: %f" %(p_t, r_t, f_t))


        with torch.no_grad():
            predicted_all = []
            gold_all = []
            for batch in test_batch:
                sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y, _ = batch
                
                model.eval()
                model_g.eval()
                #logits, cls1, cls2, new_concept_embed = model.forward(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
                
                cls1, cls2, graph = test(all_k, batch, model, k, device)
                new_con_emb = model_g.forward(cls1.shape[0], graph)

                cls_mix = torch.cat((cls1, cls2, new_con_emb), dim=1)
                logits = model_g.classification(cls_mix.type(torch.cuda.FloatTensor))

                predicted = torch.argmax(logits, -1)
                predicted = list(predicted.cpu().numpy())
                predicted_all += predicted

                gold = list(data_y.cpu().numpy())
                gold_all += gold

            print("testing results:")
            p, r, f = compute_f1(gold_all, predicted_all)
            test_prf_epoch.append([p, r, f])
            if f > best_f1: 
                best_f1 = f 
                best_precision = p 
                best_recall = r

        print("testing precision: %f, recall: %f, f1: %f" %(p, r, f))
        print("best testing precision: %f, best recall: %f, best f1: %f" %(best_precision,best_recall, best_f1))
    

