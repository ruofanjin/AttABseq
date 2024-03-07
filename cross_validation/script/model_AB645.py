# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from math import sqrt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from Radam import *
from lookahead import Lookahead
from scipy.stats import pearsonr
import os


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim * 3
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim*3, hid_dim*3)
        self.w_k = nn.Linear(hid_dim*3, hid_dim*3)
        self.w_v = nn.Linear(hid_dim*3, hid_dim*3)
        self.fc = nn.Linear(hid_dim*3, hid_dim*3)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim*3 // n_heads])).to(device) # hid_dim//n_heads = 32

    def forward(self, query, key, value, mask=None):

        bsz = query.shape[0]
        # query = key = value = [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, sent len_Q, sent len_K]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.hid_dim)
        # x = [batch size, sent len_Q, hid dim] [8, 145, 768]

        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim] [8, 145, 768]

        return x


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs1 = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers1
        self.convs2 = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, 5, padding=(5-1)//2) for _ in range(self.n_layers)])   # convolutional layers2
        self.convs3 = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, 7, padding=(7-1)//2) for _ in range(self.n_layers)])   # convolutional layers3
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim * 3)

    def forward(self, protein):
        
        #protein = [batch size, protein len,protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim] 64
        #permute for convolutional layer
        conv_input1 = conv_input.permute(0, 2, 1)
        conv_input2 = conv_input.permute(0, 2, 1)
        conv_input3 = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs1):
            #pass through convolutional layer
            #conved = [batch size, 2*hid dim, protein len]
            #pass through GLU activation function
            #conved = F.glu(conv(self.dropout(conv_input)), dim=1)
            #conved = [batch size, hid dim, protein len]
            #apply residual connection / high way
            conved = (F.glu(conv(self.dropout(conv_input1)), dim=1) + conv_input1) * self.scale
            #conved = [batch size, hid dim, protein len]
            
            #set conv_input to conved for next loop iteration
            conv_input1 = conved

        for i, conv in enumerate(self.convs2):
            #pass through convolutional layer
            #conved = [batch size, 2*hid dim, protein len]
            #pass through GLU activation function
            #conved = F.glu(conv(self.dropout(conv_input)), dim=1)
            #conved = [batch size, hid dim, protein len]
            #apply residual connection / high way
            conved = (F.glu(conv(self.dropout(conv_input2)), dim=1) + conv_input2) * self.scale
            #conved = [batch size, hid dim, protein len]
            
            #set conv_input to conved for next loop iteration
            conv_input2 = conved

        for i, conv in enumerate(self.convs3):
            #pass through convolutional layer
            #conved = [batch size, 2*hid dim, protein len]
            #pass through GLU activation function
            #conved = F.glu(conv(self.dropout(conv_input)), dim=1)
            #conved = [batch size, hid dim, protein len]
            #apply residual connection / high way
            conved = (F.glu(conv(self.dropout(conv_input3)), dim=1) + conv_input3) * self.scale
            #conved = [batch size, hid dim, protein len]
            
            #set conv_input to conved for next loop iteration
            conv_input3 = conved

        conved = torch.cat((conv_input1,conv_input2,conv_input3), 1)
        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        conved = self.ln(conved)
        return conved # conved = [batch size,protein len,hid dim]


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim*3, pf_dim*3, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim*3, hid_dim*3, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        # x = [batch size, pf dim, sent len]
        x = self.fc_2(self.do(F.relu(self.fc_1(x))))
        # x = [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim*3)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, hid_dim] # encoder output
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound len]
        # src_mask = [batch size, protein len]
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))  #self-attention
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))  #inter-attention
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg

POLY_DEGREE = 3
def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)

class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim*3, hid_dim)
        self.fc_1 = nn.Linear(hid_dim, 64)
        self.fc_2 = nn.Linear(64, 16)
        self.fc_3 = nn.Linear(16, 1)
        self.gn = nn.GroupNorm(8, 64)

    def forward(self, trg, src, trg_mask=None, src_mask=None):

        # trg = [batch size, compound len, hid dim]
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)  # ag:ab
        # trg = [batch size, compound len, hid dim]  [batch, length, 256*3]
        trg = self.fc(trg)
        """Use norm to determine which atom is significant. """
        norm = F.softmax(torch.norm(trg, dim=2), dim=1)
        norm1 = torch.norm(trg, dim=2)
        # norm = [batch size,compound len]
        summ = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                summ[i, ] += v
        # sum = [batch size,hid_dim]
        #label = self.fc_1(sum)
        # label.shape=[batch size, 32]
        #label = self.fc_2(label)
        # label.shape=[batch size, 16]
        #label = self.fc_3(label)
        return summ, norm1


class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, ags_dim=40):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(ags_dim, ags_dim))
        self.init_weight()
        self.do = nn.Dropout(0.1)
        self.fc_1 = nn.Linear(256*2, 128)
        self.fc_2 = nn.Linear(128*2, 64)
        self.fc_3 = nn.Linear(64, 16)
        self.fc_4 = nn.Linear(16, 1)

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def make_masks(self, p11n, p21n, p11_max_len, p21_max_len):
        N = len(p11n)  # batch size
        p11_mask = torch.zeros((N, p11_max_len))
        p21_mask = torch.zeros((N, p21_max_len))
        for i in range(N):
            p11_mask[i, :p11n[i]] = 1
            p21_mask[i, :p21n[i]] = 1
        p11_mask = p11_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        p21_mask = p21_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return p11_mask, p21_mask

    def forward(self, ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num, itera, correct_interaction, fold):
        # compound = [batch,atom_num,atom_dim]
        # protein = [batch,protein len,protein_dim]
        ag_s_max_len = ag_s.shape[1]
        ab_s_max_len = ab_s.shape[1]
        ag_s_mask, ab_s_mask = self.make_masks(ag_s_num, ab_s_num, ag_s_max_len, ab_s_max_len)

        ag_m_s_max_len = ag_m_s.shape[1]
        ab_m_s_max_len = ab_m_s.shape[1]
        ag_m_s_mask, ab_m_s_mask = self.make_masks(ag_m_s_num, ab_m_s_num, ag_m_s_max_len, ab_m_s_max_len)

        enc_ab_s = self.encoder(ab_s)
        enc_ab_m_s = self.encoder(ab_m_s)
        # enc_protein = [batch size, protein len, hid dim]
        enc_ag_s = self.encoder(ag_s)
        enc_ag_m_s = self.encoder(ag_m_s)
        # enc_compound = [batch size, compound len, hid dim]

        ag_ab, ag_ab_norm = self.decoder(enc_ag_s, enc_ab_s, ag_s_mask, ab_s_mask)
        ab_s_mask_change = ab_s_mask.permute(0, 1, 3, 2)
        ag_s_mask_change = ag_s_mask.permute(0, 1, 3, 2)
        ab_ag, ab_ag_norm = self.decoder(enc_ab_s, enc_ag_s, ab_s_mask_change, ag_s_mask_change)

        ag_ab_m, ag_ab_m_norm = self.decoder(enc_ag_m_s, enc_ab_m_s, ag_m_s_mask, ab_m_s_mask)
        ab_m_s_mask_change = ab_m_s_mask.permute(0, 1, 3, 2)
        ag_m_s_mask_change = ag_m_s_mask.permute(0, 1, 3, 2)
        ab_ag_m, ab_ag_m_norm = self.decoder(enc_ab_m_s, enc_ag_m_s, ab_m_s_mask_change, ag_m_s_mask_change)
        # graph???

        complex_wt = self.do(F.relu(self.fc_1(torch.cat([ag_ab,ab_ag],-1))))
        complex_mut = self.do(F.relu(self.fc_1(torch.cat([ag_ab_m,ab_ag_m],-1))))
        final2 = self.do(F.relu(self.fc_2(torch.cat([complex_wt,complex_mut],-1))))
        final2 = self.do(F.relu(self.fc_3(final2)))
        final2 = self.do(F.relu(self.fc_4(final2)))
        final2 = final2.view(-1)
        
        if os.path.exists('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ag{}.txt'.format(fold, itera)):
            f1 = open('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ag{}.txt'.format(fold, itera),'a+')
            f1.write(str(correct_interaction))
            f1.write('\n')
            f1.write(str(ag_ab_norm.tolist()))
            f1.write('\n')
        else:
            f1 = open('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ag{}.txt'.format(fold, itera),'w')
            f1.write(str(correct_interaction))
            f1.write('\n')
            f1.write(str(ag_ab_norm.tolist()))
            f1.write('\n')
        
        if os.path.exists('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ab{}.txt'.format(fold, itera)):
            f2 = open('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ab{}.txt'.format(fold, itera),'a+')
            f2.write(str(correct_interaction))
            f2.write('\n')
            f2.write(str(ab_ag_norm.tolist()))
            f2.write('\n')
        else:
            f2 = open('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ab{}.txt'.format(fold, itera),'w')
            f2.write(str(correct_interaction))
            f2.write('\n')
            f2.write(str(ab_ag_norm.tolist()))
            f2.write('\n')
        
        if os.path.exists('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ag_mut{}.txt'.format(fold, itera)):
            f3 = open('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ag_mut{}.txt'.format(fold, itera),'a+')
            f3.write(str(correct_interaction))
            f3.write('\n')
            f3.write(str(ag_ab_m_norm.tolist()))
            f3.write('\n')
        else:
            f3 = open('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ag_mut{}.txt'.format(fold, itera),'w')
            f3.write(str(correct_interaction))
            f3.write('\n')
            f3.write(str(ag_ab_m_norm.tolist()))
            f3.write('\n')
        
        if os.path.exists('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ab_mut{}.txt'.format(fold, itera)):
            f4 = open('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ab_mut{}.txt'.format(fold, itera),'a+')
            f4.write(str(correct_interaction))
            f4.write('\n')
            f4.write(str(ab_ag_m_norm.tolist()))
            f4.write('\n')
        else:
            f4 = open('/home/ruofan/bcr/cnn3_cv/645analysis/fold{}/ab_mut{}.txt'.format(fold, itera),'w')
            f4.write(str(correct_interaction))
            f4.write('\n')
            f4.write(str(ab_ag_m_norm.tolist()))
            f4.write('\n')

        return final2

    def __call__(self, data, itera, fold, train=True):

        ag_s, ab_s, ag_m_s, ab_m_s, correct_interaction, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num = data

        Loss = nn.MSELoss()
        # Loss = nn.CrossEntropyLoss()
        correct_interaction = correct_interaction.to(torch.float32)
        #correct_interaction = correct_interaction.unsqueeze(1)

        if train:
            predicted_interaction = self.forward(ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num, itera, correct_interaction, fold)
            loss = Loss(predicted_interaction, correct_interaction)
            loss = loss.float()
            #correct_labels = correct_interaction.to('cpu').data.numpy()
            #predicted_labels = predicted_interaction.to('cpu').data.numpy()
            return loss, correct_interaction, predicted_interaction
            #return loss

        else:
            predicted_interaction = self.forward(ag_s, ab_s, ag_m_s, ab_m_s, ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num, itera, correct_interaction, fold)
            #correct_label = correct_interaction.to('cpu').data.numpy()
            #predicted_label = predicted_interaction.to('cpu').data.numpy()
            return correct_interaction, predicted_interaction


# zero pack
def pack(ag_s, ab_s, ag_m_s, ab_m_s, labels, device):

    ag_s_len = 0
    ab_s_len = 0
    ag_m_s_len = 0
    ab_m_s_len = 0
    N = len(labels)

    ag_s_num = []
    for ag in ag_s:
        ag_s_num.append(ag.shape[0])
        if ag.shape[0] >= ag_s_len:
            ag_s_len = ag.shape[0]
    ab_s_num = []
    for ab in ab_s:
        ab_s_num.append(ab.shape[0])
        if ab.shape[0] >= ab_s_len:
            ab_s_len = ab.shape[0]
    ag_m_s_num = []
    for agm in ag_m_s:
        ag_m_s_num.append(agm.shape[0])
        if agm.shape[0] >= ag_m_s_len:
            ag_m_s_len = agm.shape[0]
    ab_m_s_num = []
    for abm in ab_m_s:
        ab_m_s_num.append(abm.shape[0])
        if abm.shape[0] >= ab_m_s_len:
            ab_m_s_len = abm.shape[0]

    ag_s_new = torch.zeros((N, ag_s_len, 40), device=device)
    i = 0
    for ag in ag_s:
        #ag = ag.astype(float)
        ag = torch.tensor(ag)
        a_len = ag.shape[0]
        ag_s_new[i, :a_len, :] = ag
        i += 1
    ab_s_new = torch.zeros((N, ab_s_len, 40), device=device)
    i = 0
    for ab in ab_s:
        #ab = ab.astype(float)
        ab = torch.tensor(ab)
        a_len = ab.shape[0]
        ab_s_new[i, :a_len, :] = ab
        i += 1

    ag_m_s_new = torch.zeros((N, ag_m_s_len, 40), device=device)
    i = 0
    for agm in ag_m_s:
        #ag = ag.astype(float)
        agm = torch.tensor(agm)
        a_len = agm.shape[0]
        ag_m_s_new[i, :a_len, :] = agm
        i += 1
    ab_m_s_new = torch.zeros((N, ab_m_s_len, 40), device=device)
    i = 0
    for abm in ab_m_s:
        #ab = ab.astype(float)
        abm = torch.tensor(abm)
        a_len = abm.shape[0]
        ab_m_s_new[i, :a_len, :] = abm
        i += 1

    labels_new = torch.zeros(N, dtype=torch.float, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    return (ag_s_new, ab_s_new, ag_m_s_new, ab_m_s_new, labels_new,
            ag_s_num, ab_s_num, ag_m_s_num, ab_m_s_num)


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch = batch

    def train(self, dataset, device, itera, fold):
        self.model.train()
        N = len(dataset)
        i = 0
        iteration = 0
        self.optimizer.zero_grad()
        ag_s, ab_s, ag_m_s, ab_m_s, labels = [], [], [], [], []
        train_correct_fold = torch.zeros((1,0), device=device)
        train_predict_fold = torch.zeros((1,0), device=device)
        lo = []
        for data in dataset:
            i = i+1
            ag, ab, agm, abm, label = data
            ag_s.append(ag)
            ab_s.append(ab)
            ag_m_s.append(agm)
            ab_m_s.append(abm)
            labels.append(label)
            correct_labels = torch.zeros((1,0), device=device)
            predicted_labels = torch.zeros((1,0), device=device)
            if i % self.batch == 0 or i == N:
                iteration += 1
                data_pack = pack(ag_s, ab_s, ag_m_s, ab_m_s, labels, device)
                loss, correct, predicted = self.model(data_pack, itera, fold)  # predictor.train()
                correct = correct.view(1,-1)
                predicted = predicted.view(1,-1)
                correct_labels = torch.cat([correct_labels,correct], dim=-1)
                predicted_labels = torch.cat([predicted_labels,predicted], dim=-1)
                train_correct_fold = torch.cat([train_correct_fold,correct], dim=-1)
                train_predict_fold = torch.cat([train_predict_fold,predicted], dim=-1)
                lo.append(loss)
                # loss = loss / self.batch
                loss.backward()
                
                #for name, weight in self.model.named_parameters():
                    # print("weight:", weight)
                    #if weight.requires_grad:
                        # print("weight:", weight.grad)
                        #print("weight.grad:", weight.grad.min(), weight.grad.max())  #weight.grad.mean(), 

                ag_s, ab_s, ag_m_s, ab_m_s, labels = [], [], [], [], []
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                continue
            
        Loss = nn.MSELoss()
        loss_train_1 = Loss(train_correct_fold, train_predict_fold)
        loss_train = sum(lo)/iteration
        return loss_train, train_correct_fold, train_predict_fold


def get_corr(fake_Y, Y):
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = torch.mean(fake_Y.float()), torch.mean(Y.float())
    corr = (torch.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
        torch.sqrt(torch.sum((fake_Y - fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((Y - Y_mean) ** 2)))
    return corr


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset, itera, fold):
        self.model.eval()
        N = len(dataset)
        device = torch.device('cpu')
        T = torch.zeros((1,0), device=device)
        Y = torch.zeros((1,0), device=device)
        with torch.no_grad():
            for data in dataset:
                ag_s, ab_s, ag_m_s, ab_m_s, labels = [], [], [], [], []
                ag, ab, agm, abm, label = data
                ag_s.append(ag)
                ab_s.append(ab)
                ag_m_s.append(agm)
                ab_m_s.append(abm)
                labels.append(label)
                data = pack(ag_s, ab_s, ag_m_s, ab_m_s, labels, self.model.device)
                correct, predicted = self.model(data, itera, fold, train=False)
                correct = correct.view(1,-1)
                predicted = predicted.view(1,-1)
                T = torch.cat([T,correct], dim=-1)
                Y = torch.cat([Y,predicted], dim=-1)
        T = T.squeeze()
        Y = Y.squeeze()
        print('true:', T)
        print('predict:', Y)
        pccs = get_corr(T,Y)  # an epoch's val's pccs
        mae = mean_absolute_error(T, Y)
        mse = mean_squared_error(T, Y)
        rmse = sqrt(mean_squared_error(T, Y))
        r2 = r2_score(T, Y)
        Loss = nn.MSELoss()
        loss = Loss(T, Y)  # an epoch's val's loss
        return pccs, mae, mse, rmse, r2, loss, T, Y

    def save_pccs(self,pccs,filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str,pccs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

