#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import imp
import time
import argparse
from unittest import result
import numpy as np
import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI


from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
from anomaly_detect import prune_high_fluctuation_nodes, prune_edge, anomaly_embedding_edge, gnn_prune_low_entropy_nodes,prune_anomaly_nodes_edges
from help_funcs import prune_edge_isolated
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--bkd_model', type=str, default='GTA',
                    help='bkd_model',
                    choices=['clean','UGBA','GTA','SBA_Samp','SBA_Rand'])
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Pubmed', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='tirgger_size')
parser.add_argument('--vs_size', type=int, default=40,
                    help="ratio of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="MAD",
                    choices=['prune', 'isolate', 'none','MAD_feature','MAD_embedding','MAD_confidence','MAD'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.2,
                    help="Threshold of prunning edges")
parser.add_argument('--fluctuation_thrd', type=float, default=0.003,help="Feature anomaly detection threshold")
parser.add_argument('--dist_thrd', type=float, default=0.003,help="Embedding anomaly detection threshold")
parser.add_argument('--defense_thrd', type=float, default=3,help="Predicted probability anomaly detection threshold")
parser.add_argument('--homo_loss_weight', type=float, default=0,
                    help="Weight of optimize similarity loss")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='none',
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='overall',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
# args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

import os
import sys
import logging
import datetime
dataset_name = args.dataset

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

log_dir = os.path.join('logs',args.bkd_model,args.defense_mode, dataset_name)
log_file = os.path.join(log_dir, f'{current_time}.log')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(filename=log_file, level=logging.INFO)

class Logger(object):
    def __init__(self, filename=log_file):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()
sys.stderr = Logger()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'Reddit2'):
    dataset = Reddit2(root='./data/Reddit2/', \
                    transform=transform)
elif(args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
#%% 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


# In[9]:
from models.GTA import Backdoor
import heuristic_selection as hs

import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_idx_clean_and_trigger(x, ori_x):
    num_clean_nodes = ori_x.size(0)
    num_total_nodes = x.size(0)

    idx_clean = torch.arange(num_clean_nodes, device=x.device)

    idx_trigger = torch.arange(num_clean_nodes, num_total_nodes, device=x.device)

    return idx_clean, idx_trigger

from Detect_encoder import cluster_encoder_detect,prune_anomalous_edges_by_class


def calculate_entropy(features):
    features_min = features.min(dim=1, keepdim=True)[0]
    features_max = features.max(dim=1, keepdim=True)[0]
    features_normalized = (features - features_min) / (features_max - features_min + 1e-10)
    row_sums = features_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    probabilities = features_normalized / row_sums

    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    return entropy

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
size = args.vs_size #int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print("#Attach Nodes:{}".format(size))
# here is randomly select poison nodes from unlabeled nodes
from models.construct import model_construct

if(args.selection_method == 'none'):
    idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
elif(args.selection_method == 'cluster'):
    idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif(args.selection_method == 'cluster_degree'):
    idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    idx_attach = torch.LongTensor(idx_attach).to(device)

# In[10]:
# train trigger generator
model = Backdoor(args,device)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()

idx_clean, idx_trigger = get_idx_clean_and_trigger(poison_x, data.x)

from Detect_encoder import cluster_encoder_detect, prune_anomalous_edges_by_class

if(args.defense_mode == 'prune'):
    poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'isolate'):
    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
elif (args.defense_mode == 'MAD_feature'):
    poison_edge_index,poison_edge_weights = prune_high_fluctuation_nodes(args, poison_edge_index, poison_edge_weights, poison_x, device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
elif(args.defense_mode == 'MAD_embedding'):
    encoder_x, y_pred ,gcn_encoder = cluster_encoder_detect(args, poison_x, poison_labels, poison_edge_index, device, idx_attach, idx_trigger, current_time)
    poison_edge_index, poison_edge_weights = prune_edge(args, poison_edge_index,poison_edge_weights, encoder_x, device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
elif(args.defense_mode == 'MAD_confidence'):
    poison_edge_index, poison_edge_weights ,gnn_model ,prediction_entropy_threshold = gnn_prune_low_entropy_nodes(args, poison_edge_index, poison_edge_weights,
                                                                                      poison_x, poison_labels,device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'MAD'):
    encoder_x, y_pred, gcn_encoder = cluster_encoder_detect(args, poison_x, poison_labels, poison_edge_index, device, idx_attach, idx_trigger, current_time)
    poison_edge_index, poison_edge_weights, gnn_model, prediction_entropy_threshold = prune_anomaly_nodes_edges(
        args, poison_edge_index, poison_edge_weights,poison_x,poison_labels,encoder_x, y_pred,device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
print("precent of left attach nodes: {:.3f}"\
    .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))

import torch.nn.functional as F

def calculate_std(features):
    std = torch.std(features, dim=1)
    return std

def calculate_entropy_predict(probabilities):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
#%%
# models = ['GCN','GAT', 'GraphSage']
models = ['GAT']
total_overall_asr = 0
total_overall_ca = 0
for test_model in models:
    args.test_model = test_model
    result_asr = []
    result_acc = []
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=5)
    overall_asr = 0
    overall_ca = 0
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        args.seed = seed
        # %%
        test_model = model_construct(args,args.test_model,data,device).to(device)
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)

        output = test_model(poison_x,poison_edge_index,poison_edge_weights)

        # probabilities = F.softmax(output, dim=1)
        # predicted_labels = probabilities.argmax(dim=1)
        # encoder_x = test_model.get_h(poison_x, poison_edge_index).clone().detach()
        # visualize_embedding(args, encoder_x, predicted_labels, idx_attach, idx_trigger, current_time)


        train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
        print("target class rate on Vs: {:.4f}".format(train_attach_rate))
        torch.cuda.empty_cache()
        #%%
        induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

        # clean_idx = idx_clean_test[(data.y[idx_clean_test] == args.target_class).nonzero().flatten()]
        # clean_probabilities = F.softmax(output[clean_idx], dim=1)
        # clean_entropy = calculate_entropy_predict(clean_probabilities).detach().cpu().numpy()

        # test_model = test_model.cpu()

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
        # %% inject trigger on attack test nodes (idx_atk)'''
        induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
        induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
        # do pruning in test datas'''
        if (args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
            induct_edge_index, induct_edge_weights = prune_unrelated_edge(args, induct_edge_index, induct_edge_weights,
                                                                          induct_x, device)
        elif (args.defense_mode == 'MAD_feature'):
            induct_edge_index, induct_edge_weights = prune_high_fluctuation_nodes(args, induct_edge_index,
                                                                                  induct_edge_weights, induct_x, device)
        elif (args.defense_mode == 'MAD_embedding'):
            encoder_x_test = gcn_encoder.get_h(induct_x, induct_edge_index,
                                               induct_edge_weights).clone().detach()
            y_pred_test = gcn_encoder.test(induct_x, induct_edge_index, induct_edge_weights)
            induct_edge_index, induct_edge_weights = prune_edge(args, induct_edge_index, induct_edge_weights,
                                                                encoder_x_test, device)
        elif (args.defense_mode == 'MAD_confidence'):
            with torch.no_grad():
                output_defense = gnn_model(induct_x, induct_edge_index, induct_edge_weights)
                probabilities = F.softmax(output_defense, dim=1)
                predicted_labels = probabilities.argmax(dim=1)

            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            mask = (entropy < prediction_entropy_threshold)  # & target_class_mask
            keep_edges_mask = ~(mask[induct_edge_index[0]] | mask[induct_edge_index[1]])
            # Filter the edge_index by the edges we want to keep
            induct_edge_index = induct_edge_index[:, keep_edges_mask]
            # Filter the edge weights similarly
            induct_edge_weights = induct_edge_weights[keep_edges_mask]
        elif (args.defense_mode == 'MAD'):
            with torch.no_grad():
                output_defense = gnn_model(induct_x, induct_edge_index, induct_edge_weights)
                probabilities = F.softmax(output_defense, dim=1)
                # predicted_labels = probabilities.argmax(dim=1)
            feature_std = calculate_std(induct_x)
            high_feature_std_mask = feature_std > args.fluctuation_thrd
            encoder_x_test = gcn_encoder.get_h(induct_x, induct_edge_index,
                                               induct_edge_weights).clone().detach()
            y_pred_test = gcn_encoder.test(induct_x, induct_edge_index, induct_edge_weights)
            high_dist_edges_mask = anomaly_embedding_edge(args, induct_edge_index, induct_edge_weights, encoder_x_test,
                                                          device)

            prediction_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            # target_class_mask = (predicted_labels == most_common_class)
            low_prediction_entropy_mask = (prediction_entropy < prediction_entropy_threshold)  # & target_class_mask

            keep_edges_mask = ~(
                    low_prediction_entropy_mask[induct_edge_index[0]] | high_feature_std_mask[induct_edge_index[0]] |
                    low_prediction_entropy_mask[induct_edge_index[1]] | high_feature_std_mask[induct_edge_index[1]])
            keep_edges_mask = keep_edges_mask.to(device)  #
            high_dist_edges_mask = high_dist_edges_mask.to(device)  #
            final_edges_mask = keep_edges_mask | high_dist_edges_mask
            induct_edge_index = induct_edge_index[:, final_edges_mask]
            # Filter the edge weights similarly
            induct_edge_weights = induct_edge_weights[final_edges_mask]
        # attack evaluation

        # test_model = test_model.to(device)
        output = test_model(induct_x,induct_edge_index,induct_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
        print("ASR: {:.4f}".format(train_attach_rate))
        asr = train_attach_rate
        flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
        flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
        print("Flip ASR: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
        ca = test_model.test(induct_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        print("CA: {:.4f}".format(ca))

        result_asr.append(float(flip_asr))
        result_acc.append(float(clean_acc))

    print('The final ASR:{:.5f}, {:.5f}, Accuracy:{:.5f}, {:.5f}'\
                .format(np.average(result_asr),np.std(result_asr),np.average(result_acc),np.std(result_acc)))
    overall_asr = np.average(result_asr)
    overall_ca = np.average(result_acc)
    total_overall_asr += overall_asr
    total_overall_ca += overall_ca
    test_model.to(torch.device('cpu'))
    torch.cuda.empty_cache()
total_overall_asr = total_overall_asr/len(models)
total_overall_ca = total_overall_ca/len(models)
print("Total Overall ASR: {:.4f} ".format(total_overall_asr))
print("Total Clean Accuracy: {:.4f}".format(total_overall_ca))