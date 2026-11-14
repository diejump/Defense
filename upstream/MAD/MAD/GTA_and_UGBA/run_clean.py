
#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import imp
import time
import argparse
import numpy as np
import torch

from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI


# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated,gnn_prune_unrelated_edge,prune_high_entropy_edges,prune_entropy_edges

from help_funcs import prune_edge,prune_edge_isolated
import scipy.sparse as sp

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--bkd_model', type=str, default='clean',
                    help='bkd_model',
                    choices=['clean','UGBA','GTA','SBA_Samp','SBA_Rand'])
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='ogbn-arxiv', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
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
parser.add_argument('--use_vs_number', action='store_true', default=False,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=0,
                    help="number of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="prune",
                    choices=['prune', 'isolate', 'none','entropy_feature','entropy_predict','entropy','encoder','cmp_encoder'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8,
                    help="Threshold of prunning edges")
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--defense_thrd', type=float, default=3)
parser.add_argument('--feature_thrd', type=float, default=5)
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=100,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.8,
                    help="Threshold of increase similarity")
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
args = parser.parse_args()
# args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

import os
import sys
import logging
import datetime
dataset_name = args.dataset

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

log_dir = os.path.join('logs', args.bkd_model,args.defense_mode, dataset_name)
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


from Detect_encoder import cluster_encoder_detect,multi_stage_anomaly_detection,prune_anomalous_edges_by_class

def calculate_entropy(features):
    features_min = features.min(dim=1, keepdim=True)[0]
    features_max = features.max(dim=1, keepdim=True)[0]
    features_normalized = (features - features_min) / (features_max - features_min + 1e-10)
    row_sums = features_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    probabilities = features_normalized / row_sums

    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
    return entropy

from sklearn_extra import cluster
from models.backdoor import Backdoor
from models.construct import model_construct
import heuristic_selection as hs
import matplotlib.pyplot as plt

# from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()

models = ['GCN','GAT', 'GraphSage']
total_overall_asr = 0
total_overall_ca = 0

idx_attach=idx_atk
idx_trigger=idx_atk

for test_model in models:
    args.test_model = test_model
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000,size=5)
    # seeds = [args.seed]
    overall_asr = 0
    overall_ca = 0
    for seed in seeds:
        args.seed = seed

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print(args)
        #%%
        test_model = model_construct(args,args.test_model,data,device).to(device) 
        # TODO: add multiple time seeds
        if(args.defense_mode == 'prune'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            train_edge_index, train_edge_weights = prune_unrelated_edge(args,train_edge_index,train_edge_weights,data.x,device,large_graph=False)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
        elif(args.defense_mode == 'isolate'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            train_edge_index, train_edge_weights, rel_nodes = prune_unrelated_edge_isolated(args,train_edge_index,train_edge_weights,data.x,device,large_graph=False)
            idx_tn_nodes = torch.LongTensor(list(set(idx_train) - set(rel_nodes))).to(device)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
        elif (args.defense_mode == 'entropy_feature'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            train_edge_index, train_edge_weights = prune_high_entropy_edges(args, train_edge_index, train_edge_weights, data.x, device, large_graph=False)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=False)
        elif (args.defense_mode == 'entropy_predict'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            train_edge_index, train_edge_weights, gnn_model, prediction_entropy_threshold,most_common_class = gnn_prune_unrelated_edge(args, train_edge_index, train_edge_weights, data.x, data.y, device, large_graph=False)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_train, idx_val,train_iters=args.epochs, verbose=False)
        elif (args.defense_mode == 'entropy'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            train_edge_index, train_edge_weights, gnn_model, prediction_entropy_threshold, feature_entropy_threshold,most_common_class = prune_entropy_edges(args, train_edge_index, train_edge_weights, data.x, data.y, device, large_graph=False)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=False)
        elif (args.defense_mode == 'encoder'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            encoder_x, y_pred, gcn_encoder = cluster_encoder_detect(args, data.x, data.y, train_edge_index,
                                                                    device, idx_attach, idx_trigger, current_time)
            anomalies = multi_stage_anomaly_detection(encoder_x, y_pred, global_contamination=0.1,
                                                      category_contamination=0.01, n_estimators=100)
            # poison_edge_index, poison_edge_weights  = prune_anomalous_edges_by_class (args, poison_edge_index, poison_edge_weights, anomalies, device, large_graph=False)
            # bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
            train_edge_index, train_edge_weights = prune_edge(args, train_edge_index, train_edge_weights, encoder_x,
                                                                device, large_graph=False)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=False)
            # poison_edge_index, poison_edge_weights, rel_nodes = prune_edge_isolated(args, poison_edge_index,poison_edge_weights, encoder_x, device, large_graph=False)
            # bkd_tn_nodes = torch.cat([idx_train, idx_attach]).tolist()
            # bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
        elif (args.defense_mode == 'cmp_encoder'):
            train_edge_weights = torch.ones([train_edge_index.shape[1]]).to(device)
            entropy_x = calculate_entropy(data.x).detach().cpu().numpy()
            num_nodes = data.x.size(0)
            dyad_counts, triad_counts, core_numbers = calculate_graph_structures(train_edge_index, num_nodes)
            augmented_x = augment_features_with_graph_properties(data.x, entropy_x, dyad_counts, triad_counts,
                                                                 core_numbers, device)
            encoder_x, y_pred, gcn_encoder = cmp_encoder_detect(args, augmented_x, data.y, train_edge_index,
                                                                    device, idx_attach, idx_trigger, current_time)
            # anomalies = multi_stage_anomaly_detection(encoder_x, y_pred, global_contamination=0.1,
            #                                           category_contamination=0.01, n_estimators=100)
            # poison_edge_index, poison_edge_weights  = prune_anomalous_edges_by_class (args, poison_edge_index, poison_edge_weights, anomalies, device, large_graph=False)
            # bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
            train_edge_index, train_edge_weights = prune_edge(args, train_edge_index, train_edge_weights, encoder_x,
                                                                device, large_graph=False)
            test_model.fit(data.x, train_edge_index, train_edge_weights, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=False)
            # poison_edge_index, poison_edge_weights, rel_nodes = prune_edge_isolated(args, poison_edge_index,poison_edge_weights, encoder_x, device, large_graph=False)
            # bkd_tn_nodes = torch.cat([idx_train, idx_attach]).tolist()
            # bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
        else:
            test_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
        output = test_model(data.x,data.edge_index,None)
        train_attach_rate = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean()
        # print("train_attach_rate:",train_attach_rate)
        print("ASR: {:.4f}".format(train_attach_rate))
        asr = train_attach_rate
        flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
        flip_asr = (output.argmax(dim=1)[flip_idx_atk] == args.target_class).float().mean()
        print("Flip ASR: {:.4f}/{} nodes".format(flip_asr, flip_idx_atk.shape[0]))
        # print("target class rate on Vs: {:.4f}".format(train_attach_rate))
        # torch.cuda.empty_cache()
        #%%
        # induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
        # induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
        if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
            induct_edge_weights = torch.ones([data.edge_index.shape[1]]).to(device)
            induct_edge_index, induct_edge_weights = prune_unrelated_edge(args,data.edge_index,induct_edge_weights,data.x,device,large_graph=False)
            clean_acc = test_model.test(data.x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)
        else:
            clean_acc = test_model.test(data.x,data.edge_index,None,data.y,idx_clean_test)
        test_model = test_model.cpu()

        print("accuracy on clean test nodes: {:.4f} ({} model, Seed: {})".format(clean_acc,args.test_model, args.seed))
        overall_ca += clean_acc
        overall_asr += flip_asr

    overall_ca = overall_ca/len(seeds)
    overall_asr = overall_asr / len(seeds)
    print("Overall ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, args.test_model, args.seed))
    print("Overall Clean Accuracy: {:.4f} ({} model)".format(overall_ca, args.test_model))

    total_overall_asr += overall_asr
    total_overall_ca += overall_ca
    test_model.to(torch.device('cpu'))
    torch.cuda.empty_cache()
total_overall_asr = total_overall_asr/len(models)
total_overall_ca = total_overall_ca/len(models)
print("Total Overall ASR: {:.4f} ".format(total_overall_asr))
print("Total Clean Accuracy: {:.4f}".format(total_overall_ca))