import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import palettable
import torch



def idx(file):
    f = open(file)
    line = f.readlines()[1]
    line = line.replace('[','')
    line = line.replace(']','')
    line = line.replace('\n','')
    line = line.split(', ')
    line = [int(i) for i in line]
    return line

def seqs(ls,file):
    f = pd.read_csv(file)
    ab = f['a'].tolist()
    ag = f['b'].tolist()
    abm = f['a_mut'].tolist()
    agm = f['b_mut'].tolist()
    ls_pdb = []
    ls_mut = []
    ls_ab = []
    ls_ag = []
    ls_abm = []
    ls_agm = []
    ls_label = []
    for i in ls:
        ls_pdb.append(f['PDB'].tolist()[i])
        ls_mut.append(f['mutation'].tolist()[i])
        ls_ab.append(ab[i])
        ls_ag.append(ag[i])
        ls_abm.append(abm[i])
        ls_agm.append(agm[i])
        ls_label.append(float(f['ddG'].tolist()[i]))
    return ls_pdb, ls_mut, ls_ab,ls_ag,ls_abm,ls_agm, ls_label

def heatmap_make(input,path):
    x = input
    plt.figure(figsize=(121, 99)) # ,dpi=100
    ax = sns.heatmap(data=x, annot=True, fmt=".2f", vmin=0, vmax=1,
                     cmap='RdBu_r', center=None, cbar=True, # palettable.cmocean.diverging.Curl_10.mpl_colors
                     linewidths=0.5,linecolor='white',
                     cbar_kws={# 'label': 'norm number', # color bar的名称
                               'orientation': 'vertical', # color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                               'ticks':np.arange(0,1,0.2), # color bar中刻度值范围和间隔
                               'format':'%.3f', # 格式化输出color bar中刻度值
                               'pad':0.05, # color bar与热图之间距离，距离变大热图会被压缩
                               },
                               ) # mask=x<0.6
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=100) # ax.tick_params(right=True, top=True, labelright=True, labeltop=True)
    sns.despine(top=True, right=True, left=False, bottom=False) # 坐標軸顯示邊框黑綫
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.colorbar()
    plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1.0) # 自动调整子图参数，使之填充整个图像区域。这是个实验特性，可能在一些情况下不工作。它仅仅检查坐标轴标签、刻度标签以及标题的部分
    plt.savefig(path)
    return

def data_normal(data):
    d_min = data.min()
    if d_min < 0:
        data += torch.abs(d_min)
        d_min = data.min()
    d_max = data.max()
    dst = d_max-d_min
    norm_data = (data-d_min).true_divide(dst)
    return norm_data

def data_normal_2d(orign_data,dim="col"):  # (-1, 1)
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])  
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    return norm_data

def check_pos(protype,seq1,seq2):
    pos = 0
    if len(seq1)==len(seq2):
        for i in range(len(seq1)):
            if seq1[i]==seq2[i]:
                continue
            else:
                pos = '{}{}_{}{}'.format(protype,seq1[i],i,seq2[i])
                i = len(seq1)-1
    else:
        pos = 0
    return pos


dir = '/home/ruofan/bcr/split/interpretable/1131'
pdbidx = idx(dir+'/split.txt')
ls_pdb, ls_mut, ls_ab,ls_ag,ls_abm,ls_agm, ls_label = seqs(pdbidx,dir+'/S1131.csv')
count = len(ls_label)
ab = open('ab15.txt').readlines()[-2*count:]
ag = open('ag15.txt').readlines()[-2*count:]
abm = open('ab_mut15.txt').readlines()[-2*count:]
agm = open('ag_mut15.txt').readlines()[-2*count:]
ab_inters = []
ag_inters = []
abm_inters = []
agm_inters = []
for i in range(2*count):
    if i%2==1:
        ab[i] = ab[i].replace('[[','')
        ab[i] = ab[i].replace(']]','')
        ab[i] = [float(j) for j in ab[i].split(', ')]
        ab_inters.append(ab[i])
        ag[i] = ag[i].replace('[[','')
        ag[i] = ag[i].replace(']]','')
        ag[i] = [float(j) for j in ag[i].split(', ')]
        ag_inters.append(ag[i])
        abm[i] = abm[i].replace('[[','')
        abm[i] = abm[i].replace(']]','')
        abm[i] = [float(j) for j in abm[i].split(', ')]
        abm_inters.append(abm[i])
        agm[i] = agm[i].replace('[[','')
        agm[i] = agm[i].replace(']]','')
        agm[i] = [float(j) for j in agm[i].split(', ')]
        agm_inters.append(agm[i])

inter = pd.DataFrame({'pdb':ls_pdb,'mutation':ls_mut,
                      'ab_seq':ls_ab,'ag_seq':ls_ag,'abm_seq':ls_abm,'agm_seq':ls_agm,
                      'ab_inter':ab_inters,'ag_inter':ag_inters,'abm_inter':abm_inters,'agm_inter':agm_inters,
                      'label':ls_label})
inter.to_csv(dir+'/interpre.csv')

count = len(ls_label)
for idx in range(count):
    print('pdb name:',ls_pdb[idx])
    name = ls_pdb[idx]
    print('mutation:',ls_mut[idx])
    stru_mutation = ls_mut[idx].replace(':','_')
    ab_seq = ls_ab[idx]
    ab_inter = np.array(ab_inters[idx]).reshape(len(ab_inters[idx]),1)
    ag_seq = ls_ag[idx]
    ag_inter = np.array(ag_inters[idx]).reshape(1,len(ag_inters[idx]))
    abm_seq = ls_abm[idx]
    abm_inter = np.array(abm_inters[idx]).reshape(len(abm_inters[idx]),1)
    agm_seq = ls_agm[idx]
    agm_inter = np.array(agm_inters[idx]).reshape(1,len(agm_inters[idx]))
    if check_pos('ab',ab_seq,abm_seq)==0:
        seq_mutation = check_pos('ag',ag_seq,agm_seq)
    else:
        seq_mutation = check_pos('ab',ab_seq,abm_seq)
    print('seq mutation:',seq_mutation)
    # print(ab_inter.shape,ag_inter.shape,abm_inter.shape,agm_inter.shape)
    ab_ag = np.matmul(ab_inter, ag_inter)
    abm_agm = np.matmul(abm_inter,agm_inter)
    ab_ag = data_normal(torch.Tensor(ab_ag))
    abm_agm = data_normal(torch.Tensor(abm_agm))
    distinct = data_normal(torch.Tensor(abm_agm-ab_ag))
    distinct = distinct.tolist()
    ab_ag = ab_ag.tolist()
    abm_agm = abm_agm.tolist()
    # print(ab_ag.shape,abm_agm.shape)
    ab_ag_csv = pd.DataFrame(data=ab_ag, columns=list(ag_seq), index=list(ab_seq))
    abm_agm_csv = pd.DataFrame(data=abm_agm, columns=list(agm_seq), index=list(abm_seq))
    ab_ag_csv.to_csv(dir+'/interpre_csv'+'/ab_ag-{}-{}-{}-{}.csv'.format(name, stru_mutation, seq_mutation, idx))
    abm_agm_csv.to_csv(dir+'/interpre_csv'+'/abm_agm-{}-{}-{}-{}.csv'.format(name, stru_mutation, seq_mutation, idx))
    heatmap_make(ab_ag_csv, dir+'/interpre_heatmap'+'/ab-{}-{}-{}-{}.png'.format(name, stru_mutation, seq_mutation, idx))  # 太抽象了，特别小，数据量大的时候真的不合适。
    heatmap_make(abm_agm_csv, dir+'/interpre_heatmap'+'/abm-{}-{}-{}-{}.png'.format(name, stru_mutation, seq_mutation, idx))
    
    
    # print(distinct.shape)
    distinct_csv = pd.DataFrame(data=distinct, columns=list(agm_seq), index=list(abm_seq))
    distinct_csv.to_csv(dir+'/interpre_csv'+'/dis-{}-{}-{}-{}.csv'.format(name, stru_mutation, seq_mutation, idx))
    heatmap_make(distinct_csv, dir+'/interpre_heatmap'+'/dis-{}-{}-{}-{}.png'.format(name, stru_mutation, seq_mutation, idx))
