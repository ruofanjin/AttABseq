# AttABseq: An Attention-based Deep Learning Prediction Method for Antigen-Antibody Binding Affinity Changes Based on Protein Sequences

## Introduction

AttABseq is an end-to-end sequence-based deep learning model for the predictions of the antigen-antibody binding affinity changes connected with antibody mutations.

## Files Architecture
```
AttABseq
├── analysis
├── attention_ablation
│   ├── k-cv_no-attention
│   │   ├── 645analysis
│   │   ├── 1101analysis
│   │   ├── 1131analysis
│   │   ├── data
│   │   │   ├──AB645.csv
│   │   │   ├──AB645order.csv
│   │   │   ├──AB1101.csv
│   │   │   ├──AB1101order.csv
│   │   │   ├──S1131.csv
│   │   │   └──S1131order.csv
│   │   ├── ncbi-blast-2.12.0+
│   │   ├── output645
│   │   │   ├──best_pcc_model
│   │   │   ├──best_pcc_result
│   │   │   ├──best_r2_model
│   │   │   ├──best_r2_result
│   │   │   ├──loss_min_model
│   │   │   ├──loss_min_result
│   │   ├── output1101
│   │   ├── output1131
│   │   ├── script
│   │   │   ├──main_AB645.py
│   │   │   ├──main_AB1101.py
│   │   │   ├──main_S1131.py
│   │   │   ├──model_AB645.py
│   │   │   ├──model_AB1101.py
│   │   │   ├──model_S1131.py
│   │   │   ├──predict.py
│   │   │   ├──lookahead.py
│   │   │   ├──Radam.py
│   │   │   └──pytorchtools.py
│   └── split_no-attention
├── cross_validation
│   ├── 645analysis
│   ├── 1101analysis
│   ├── 1131analysis
│   ├── data
│   ├── ncbi-blast-2.12.0+
│   ├── output645
│   ├── output1101
│   ├── output1131
│   └── script
├── split
│   ├── 645analysis
│   ├── 1101analysis
│   ├── 1131analysis
│   ├── data
│   ├── ncbi-blast-2.12.0+
│   ├── output645
│   ├── output1101
│   ├── output1131
│   └── script
├── interpretability
│   ├── scatter.py
│   ├── 645
│   │   ├── AttABseq_split_645.csv
│   │   ├── scatter.py
│   │   └── split-645.png
│   ├── 1101
│   ├── 1131
│   ├── interpretable
│   │   ├── 645
│   │   │   ├── interpre_csv
│   │   │   ├── interpre_heatmap
│   │   │   ├── 645_interpre.csv
│   │   │   ├── ab16.txt
│   │   │   ├── ab_mut16.txt
│   │   │   ├── ag16.txt
│   │   │   ├── ag_mut16.txt
│   │   │   ├── AB645.csv
│   │   │   ├── interpre.csv
│   │   │   ├── interpretable.csv
│   │   │   ├── interpre.py
│   │   │   └── split.txt
│   │   ├── 1101
│   │   └── 1131
```
## Usage

### 1. Environment
- python 3.7.0
- pytorch 1.7.0
- torchvision 0.8.0
- numpy 1.21.5
- pandas 1.3.5
- scikit-learn 1.0.2
- scipy 1.7.3
- seaborn 0.12.2
- matplotlib 3.5.3
- networkx 2.6.3
- xarray 0.20.2

### 2. Data
- k-cv: AB645.csv / AB1101.csv / S1131.csv
- label-ascending ordered split: AB645order.csv / AB1101order.csv / S1131order.csv

### 3. Training
```
conda activate yourenvironment
python main.py
```
You can find your results in the folder "output".

### 4.  Testing
```
conda activate yourenvironment
python predict.py
```
You can find your results in the folder "output".

### 5. Interpretable
```
conda activate yourenvironment
python interpre.py
```
You can find your results in the folder "interpre_csv" & "interpre_heatmap".

