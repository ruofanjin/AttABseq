# -*- coding: utf-8 -*-
import pandas as pd
import torch
from numpy import *
import numpy as np
import random
import time
from model import *
import timeit
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
#from data import diction,dataset
import os
import argparse



def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""

    dir_input = ('your data')
    antibodies = np.load('your pssm numpy file',allow_pickle=True)
    antibodies_mut = np.load('your pssm numpy file',allow_pickle=True)
    antigens = np.load('your pssm numpy file',allow_pickle=True)
    antigens_mut = np.load('your pssm numpy file',allow_pickle=True)
    interactions = np.load('your pssm numpy file',allow_pickle=True)


    """ create model ,trainer and tester """
    antibody_dim = 20
    # protein_dim = 100
    antigen_dim = 20
    # atom_dim = 34
    hid_dim = 256
    n_layers = 3  #3
    n_heads = 8
    pf_dim = 64
    dropout = 0.1
    batch = 8  # 原64
    lr = 0.00001
    weight_decay = 1e-4
    decay_interval = 5  # 每5轮观察一次，判断是否改变lr。
    lr_decay = 1  # 之前从1改到了0.9。但是我需要对学习率作一个动态变化。
    iteration = 100
    kernel_size = 7  # 7


    encoder = Encoder(antibody_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(antigen_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

    model = Predictor(encoder, decoder, device)
    model.load_state_dict(torch.load("model"))
    model.to(device)
    #if is_distributed:
        #print("start init process group")
        # device_ids will include all GPU devices by default
        #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        #print("end init process group")
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)


    """Start training."""
    print('Training...')

    best_pearson = 0
    
    

    """Output files."""
    file_PCCS = '../output/result/RECORD.txt'
    file_model = '../output/model/model'
    #PCCS = ('Epoch\tTime(sec)\tLoss_train\tLoss_val\tpearson')
    PCCS = ('Epoch\tTime(sec)\tLoss_val\tpearson')
    print(PCCS)

    with open(file_PCCS, 'w') as f:
        f.write(PCCS + '\n')


    start = timeit.default_timer()

    for epoch in range(1, iteration + 1):
        
        train_idx = [i for i in range(558)]
        train_idx = np.array(train_idx)
        val_idx = [j for j in range(645)[:]]
        val_idx = np.array(val_idx)
        print(val_idx,len(val_idx))
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        
        antigens_train, antigens_val = np.array(antigens)[train_idx], np.array(antigens)[val_idx]  # ag
        antibodies_train, antibodies_val = np.array(antibodies)[train_idx], np.array(antibodies)[val_idx]  # ab
        antigens_mut_train, antigens_mut_val = np.array(antigens_mut)[train_idx], np.array(antigens_mut)[val_idx]  # ag_mut
        antibodies_mut_train, antibodies_mut_val = np.array(antibodies_mut)[train_idx], np.array(antibodies_mut)[val_idx]  # ab_mut
        interactions_train, interactions_val = np.array(interactions)[train_idx], np.array(interactions)[val_idx]  # Y
        
        dataset_train = list(zip(antigens_train, antibodies_train, antigens_mut_train, antibodies_mut_train, interactions_train))
        #dataset_train = torch.utils.data.DataLoader(dataset_train, sampler=torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=ngpus_per_node, rank=0))
        dataset_val = list(zip(antigens_val, antibodies_val, antigens_mut_val, antibodies_mut_val, interactions_val))
        #dataset_val = torch.utils.data.DataLoader(dataset_val, sampler=torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=ngpus_per_node, rank=0))

        #loss_train_fold, y_train_true, y_train_predict = trainer.train(dataset_train, device, epoch)  # numpy arrays record for an epoch loss.
        pccs_val, loss_val_fold, y_val_true, y_val_predict = tester.test(dataset_val, epoch)  # pccs_dev && loss_val_fold are for an epoch

        end = timeit.default_timer()
        time = end - start

        #PCCS = [epoch, time, loss_train_fold.tolist(), loss_val_fold.tolist(), pccs_val.tolist()]
        PCCS = [epoch, time, loss_val_fold.tolist(), pccs_val.tolist()]
        tester.save_pccs(PCCS, file_PCCS)
            
            
        #if pccs_val > best_pearson:
            #tester.save_model(model, file_model)  # 根据pearson系数的情况保存模型。
            #best_pearson = pccs_val

        print('\t'.join(map(str, PCCS)))
    #print(apple)  #SHOUDONGTIAOTING
            

        #fold_train = mean(loss_train_ls_fold)
        #fold_val = mean(loss_val_ls_fold)
        #loss_train_ls.append(fold_train)
        #loss_val_ls.append(fold_val)

    #print(loss_train_ls)
    #print(loss_val_ls)


