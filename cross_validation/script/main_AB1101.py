# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import torch
from numpy import *
import numpy as np
import random
import time
from model3 import *
import timeit
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from pytorchtools import EarlyStopping
#from data import diction,dataset


class feature(object):
    def __init__(self,seq):
        self.seq = seq
        #self.length = max

    def seq2onehot(self):
        aas = {'X':0,'A':1,'R':2,'N':3,'D':4,'C':5,
               'Q':6,'E':7,'G':8,'H':9,'I':10,
               'L':11,'K':12,'M':13,'F':14,'P':15,
               'S':16,'T':17,'W':18,'Y':19,'V':20}
        seq_onehot = np.zeros((len(self.seq),len(aas)))
        for i, aa in enumerate(self.seq[:]):
            seq_onehot[i, (aas[aa])] = 1
        #seq_onehot = ''.join(seq_onehot)
        seq_onehot = seq_onehot[:,1:]  # except X
        return seq_onehot

    def seq2pssm(self):
        with open('1101aa.fasta', 'w') as f:
            f.write('>name\n')
            f.write(self.seq)
        f.close()
        os.system('../ncbi-blast-2.12.0+/bin/psiblast -query 1101aa.fasta -db ../ncbi-blast-2.12.0+/bin/swissprot -num_iterations 3 -out 1101.txt -out_ascii_pssm 1101aa.pssm')
        with open('1101aa.pssm', 'r') as inputpssm:
            count = 0
            pssm_matrix = []
            for eachline in inputpssm:
                count += 1
                if count <= 3:
                    continue
                if not len(eachline.strip()):
                    break
                col = eachline.strip()
                col = col.split(' ')
                col = [x for x in col if x != '']
                col = col[2:22]
                col = [int(x) for x in col]
                oneline = col
                pssm_matrix.append(oneline)
            seq_pssm = np.array(pssm_matrix)
        return seq_pssm

    def seq2esm(self):  # Up in the air...
        return seq_esm

def all_feature(ls):
    all = []
    for s in ls:
        if s==s:
            f1 = feature(s).seq2onehot()
            f2 = feature(s).seq2pssm()
            f = np.concatenate((f1,f2),axis=1)
            all.append(f)
        else:
            all.append(s)  # if s!=s that f is nan
    #all = np.array(all)
    return all

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


if __name__ == "__main__":
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    csv = pd.read_csv('../data/AB1101.csv',usecols=['PDB','Mutation','antibody_light_seq','antibody_heavy_seq','antigen_a_seq','antigen_b_seq','antibody_light_seq_mut','antibody_heavy_seq_mut','antigen_a_seq_mut','antigen_b_seq_mut', 'ddG'])
    names = csv['PDB'].tolist()
    abls = csv['antibody_light_seq'].tolist()
    abhs = csv['antibody_heavy_seq'].tolist()
    agas = csv['antigen_a_seq'].tolist()
    agbs = csv['antigen_b_seq'].tolist()
    abls_m = csv['antibody_light_seq_mut'].tolist()
    abhs_m = csv['antibody_heavy_seq_mut'].tolist()
    agas_m = csv['antigen_a_seq_mut'].tolist()
    agbs_m = csv['antigen_b_seq_mut'].tolist()
    labels = csv['ddG'].tolist()

    antibodies_l = all_feature(abls)
    antibodies_h = all_feature(abhs)
    antigens_a = all_feature(agas)
    antigens_b = all_feature(agbs)
    antibodies_l_mut = all_feature(abls_m)
    antibodies_h_mut = all_feature(abhs_m)
    antigens_a_mut = all_feature(agas_m)
    antigens_b_mut = all_feature(agbs_m)

    antibodies = []  #1101
    antigens = []
    antibodies_mut = []
    antigens_mut = []
    i=0
    for i in range(len(antibodies_l)):
        if isinstance(antibodies_h[i],float):
            antibodies.append(antibodies_l[i])
        else:
            antibodies.append(np.concatenate((antibodies_l[i],antibodies_h[i]),axis=0))
        if isinstance(antigens_b[i],float):
            antigens.append(antigens_a[i])
        else:
            antigens.append(np.concatenate((antigens_a[i],antigens_b[i]),axis=0))
        if isinstance(antibodies_h_mut[i],float):
            antibodies_mut.append(antibodies_l_mut[i])
        else:
            antibodies_mut.append(np.concatenate((antibodies_l_mut[i],antibodies_h_mut[i]),axis=0))
        if isinstance(antigens_b_mut[i],float):
            antigens_mut.append(antigens_a_mut[i])
        else:
            antigens_mut.append(np.concatenate((antigens_a_mut[i],antigens_b_mut[i]),axis=0))
        i+=1
    print('i:',i)
    interactions = np.array(labels)


    """Start training."""
    print('Training...')

    n_splits = 5
    kf = KFold(n_splits=5, shuffle=True)

    i=0
    for train_index, val_index in kf.split(interactions):
        
        """ create model ,trainer and tester """
        antibody_dim = 40
        # protein_dim = 100
        antigen_dim = 40
        # atom_dim = 34
        hid_dim = 256
        n_layers = 3  #3
        n_heads = 8
        pf_dim = 64
        dropout = 0.1
        batch = 8  # 64
        lr = 0.00001
        weight_decay = 1e-4
        decay_interval = 5
        lr_decay = 1
        iteration = 150
        kernel_size = 3  # 7
        minloss = 1000
        best_pearson = -1000
        best_r2 = -1000
        
        encoder = Encoder(antibody_dim, hid_dim, n_layers, kernel_size, dropout, device)
        decoder = Decoder(antigen_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
        
        model = Predictor(encoder, decoder, device)
        # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
        model.to(device)
        #if is_distributed:
            #print("start init process group")
            # device_ids will include all GPU devices by default
            #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
            #print("end init process group")
        
        trainer = Trainer(model, lr, weight_decay, batch)
        tester = Tester(model)
        
        i+=1
        print('*************************** start training on Fold %s ***************************'%i)

        with open('/home/ruofan/bcr/cnn3_cv/1101analysis/fold{}.txt'.format(i),'w') as file_fold:
            file_fold.write(str(train_index))
            file_fold.write('\n')
            file_fold.write(str(val_index))
            file_fold.write('\n')
        file_fold.close()
        
        antigens_train, antigens_val = np.array(antigens)[train_index], np.array(antigens)[val_index]  # ag
        antibodies_train, antibodies_val = np.array(antibodies)[train_index], np.array(antibodies)[val_index]  # ab
        antigens_mut_train, antigens_mut_val = np.array(antigens_mut)[train_index], np.array(antigens_mut)[val_index]  # ag_mut
        antibodies_mut_train, antibodies_mut_val = np.array(antibodies_mut)[train_index], np.array(antibodies_mut)[val_index]  # ab_mut
        interactions_train, interactions_val = np.array(interactions)[train_index], np.array(interactions)[val_index]  # Y

        dataset_train = list(zip(antigens_train, antibodies_train, antigens_mut_train, antibodies_mut_train, interactions_train))
        #dataset_train = torch.utils.data.DataLoader(dataset_train, sampler=torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=ngpus_per_node, rank=0))
        dataset_val = list(zip(antigens_val, antibodies_val, antigens_mut_val, antibodies_mut_val, interactions_val))
        #dataset_val = torch.utils.data.DataLoader(dataset_val, sampler=torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=ngpus_per_node, rank=0))

        """Output files."""
        file_loss_min_PCCS = '../output645/loss_min_result/RECORD_{}.txt'.format(i)
        file_loss_min_model = '../output645/loss_min_model/model_{}'.format(i)
        file_best_pcc_PCCS = '../output645/best_pcc_result/RECORD_{}.txt'.format(i)
        file_best_pcc_model = '../output645/best_pcc_model/model_{}'.format(i)
        file_best_r2_PCCS = '../output645/best_r2_result/RECORD_{}.txt'.format(i)
        file_best_r2_model = '../output645/best_r2_model/model_{}'.format(i)
        
        PCCS = ('Epoch\tTime(sec)\tLoss_train\tLoss_val\tpearson\tMAE\tMSE\tRMSE\tr2')
        print(PCCS)

        with open(file_loss_min_PCCS, 'w') as f:
            f.write(PCCS + '\n')
        with open(file_best_pcc_PCCS, 'w') as f:
            f.write(PCCS + '\n')
        with open(file_best_r2_PCCS, 'w') as f:
            f.write(PCCS + '\n')

        start = timeit.default_timer()
        
        early_stopping = EarlyStopping(patience=7, verbose=True)
        for epoch in range(1, iteration + 1):

            print('Epoch:',epoch)
            loss_train_fold, y_train_true, y_train_predict = trainer.train(dataset_train, device, epoch, i)  # numpy arrays record for an epoch loss.
            pccs_val, mae_val, mse_val, rmse_val, r2_val, loss_val_fold, y_val_true, y_val_predict = tester.test(dataset_val, epoch, i)  # pccs_dev && loss_val_fold are for an epoch

            end = timeit.default_timer()
            time = end - start
            
            early_stopping(loss_val_fold.tolist(), model, file_loss_min_model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

            PCCS = [epoch, time, loss_train_fold.tolist(), loss_val_fold.tolist(), pccs_val.tolist(), mae_val, mse_val, rmse_val, r2_val]
            
            if loss_val_fold.tolist() < minloss:
                tester.save_pccs(PCCS, file_loss_min_PCCS)
                tester.save_model(model, file_loss_min_model)
                minloss = loss_val_fold.tolist()
            
            if pccs_val.tolist() > best_pearson:
                tester.save_pccs(PCCS, file_best_pcc_PCCS)
                tester.save_model(model, file_best_pcc_model)
                best_pearson = pccs_val.tolist()
            
            if r2_val > best_r2:
                tester.save_pccs(PCCS, file_best_r2_PCCS)
                tester.save_model(model, file_best_r2_model)
                best_r2 = r2_val
    
            print('\t'.join(map(str, PCCS)))
            
