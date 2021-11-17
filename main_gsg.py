import models.graphsage.train_sampling as train
import argparse
import torch as th
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm

from models.graphsage.model import SAGE
from models.graphsage.load_graph import load_reddit, inductive_split, load_ogb

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)

    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")

    # sjq
    argparser.add_argument('--enable-mask', action='store_true')
    argparser.add_argument('--enable-save', action='store_true')
    argparser.add_argument("--enable-zeros", action='store_true')
    argparser.add_argument("--enable-evaluate", action='store_true')
    argparser.add_argument("--mask-weight-decay", type=float, default=5e-3,
                           help="Weight for L2 loss")
    argparser.add_argument("--enable-training", action='store_true')

    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products')
    else:
        raise Exception('unknown dataset')

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        train_labels = val_labels = test_labels = g.ndata.pop('labels')

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
        val_nfeat, val_labels, test_nfeat, test_labels
    
    args.enable_mask = False

    for data_set in ['reddit', "ogbn-products"]:
        args.dataset = data_set
        train.run(args, device, data)

    args.enable_mask = True
    for mask in [5e-3, 1e-3, 5e-4, 1e-4, 5e-2, 1e-2]:
        for data_set in ['reddit', "ogbn-products"]:
            args.mask_weight_decay = mask
            args.dataset = data_set
            print("mask weight decay: ", mask)
            print("dataset: ", data_set)
            train.run(args, device, data)
