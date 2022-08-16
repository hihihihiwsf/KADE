import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertAdam

from dataset import Dataset
from IPython import embed

class GraphEnhanceNetworks(nn.Module):
    def __init__(self, y_num, hiddensize, concept_dim, graph_hidden_dim, graph_output_dim, pretrained_concept_emd, concept_emd, device):

        super(GraphEnhanceNetworks, self).__init__()
        self.concept_dim = concept_dim
        self.hiddensize = hiddensize
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_output_dim = graph_output_dim
        self.concept_emd = concept_emd

        self.device = device
        
        self.graph_encoder = GCN_Encoder(self.concept_dim, self.graph_hidden_dim, self.graph_output_dim,
                                         pretrained_concept_emd=pretrained_concept_emd, device=device)

        self.classification= nn.Linear(768 * 3, y_num)

    def forward(self, bsz, graph):

        if graph is not None:
            graph = graph.to(self.device)
            output_graphs = self.graph_encoder(graph)

            n_node = output_graphs.ndata["h"].shape[0]
            new_concept_embeds = torch.zeros(bsz, n_node, self.graph_output_dim).to(self.device)
            for index, g in enumerate(dgl.unbatch(output_graphs)):
                new_concept_embeds[index][:g.ndata["h"].shape[0]] = g.ndata["h"]
                

            output_concept_embeds = torch.cat((output_graphs.ndata["h"], torch.zeros(1, self.graph_output_dim).to(self.device))) # len(output_concept_embeds) as padding

            # new_concept_embed = nn.Embedding(output_concept_embeds.size()[0], output_concept_embeds.size()[1])
            # new_concept_embed.weight = nn.Parameter(output_concept_embeds)

            #new_concept_embed = torch.cat((output_graphs.ndata["h"], s_vec_batched.new_zeros((1, self.graph_output_dim))))
            #new_concept_embed = new_concept_embed.to(self.device)
            
        graph_embed = torch.sum(new_concept_embeds, dim=1).to(self.device)

        return graph_embed

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GraphConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GraphConvLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class GCN_Encoder(nn.Module):
    def __init__(self, concept_dim, hidden_dim, output_dim, pretrained_concept_emd, device, concept_emd=None):
        super(GCN_Encoder, self).__init__()

        self.gcn1 = GraphConvLayer(concept_dim, hidden_dim, F.relu)
        self.gcn2 = GraphConvLayer(hidden_dim, output_dim, F.relu)
        self.device = device

        if pretrained_concept_emd is not None and concept_emd is None:
            self.concept_emd = nn.Embedding(num_embeddings=pretrained_concept_emd.size(0),
                                        embedding_dim=pretrained_concept_emd.size(1))
            self.concept_emd.weight = nn.Parameter(pretrained_concept_emd)  # init
        elif pretrained_concept_emd is None and concept_emd is not None:
            self.concept_emd = concept_emd

    def forward(self, g):

        features = g.ndata["feat"].to(self.device) # self.concept_emd(g.ndata["f"].to(self.device))
        x = self.gcn1(g.to(self.device), features)
        x = self.gcn2(g, x)
        g.ndata['h'] = x
        return g
