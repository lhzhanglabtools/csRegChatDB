import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from scipy.stats import pearsonr
from scipy.stats import spearmanr
#load data
import numpy as np
import pandas as pd
import random
from torch.optim import Adam
import os
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
import shap
hidden_size  = 64
hidden_size2 = 16
output_size = 1
from joblib import Parallel,delayed
seed_value = 42
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self,input_size,activef):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.activef=activef
    def forward(self, x):
        #x = torch.sigmoid(self.fc1(x))
        if self.activef=='ReLU':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        if self.activef=='sigmoid':
            x = F.sigmoid(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
        if self.activef=='tanh':
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
def build_LRadj4gene(adjdir, database_filter1, gene_item):
    sub_database_filter1 = database_filter1[database_filter1['TG_Symbol']==gene_item]
    ligand_list = list(set(sub_database_filter1['Ligand_Symbol'].tolist()))
    receptor_list = list(set(sub_database_filter1['Receptor_Symbol'].tolist()))
    LR_list = ligand_list + receptor_list
    LR_df = pd.DataFrame(np.zeros((len(LR_list), len(LR_list)), dtype=float), index=LR_list, columns=LR_list)
    for item in sub_database_filter1.values:
        ligand = item[0]
        receptor = item[1]
        LR_df.loc[ligand, receptor] = 1
        LR_df.loc[receptor, ligand] = 1
    LR_df.to_csv(adjdir / (gene_item + '_LR_adj.csv'))
    return LR_df  
def build_TFREadj4gene(adjdir, database_filter1, TF_peak_union, gene_item, TG_peak, expmat,atac_mat):
    REs = list(set(TG_peak[gene_item]))
    REs = [id for id in REs if id in atac_mat.index]
    sub_database_filter1 = database_filter1[database_filter1['TG_Symbol']==gene_item]
    TFs = list(set(sub_database_filter1['TF_Symbol'].tolist()))
    TFs = [id for id in TFs if id in expmat.index]
    TFRE_list = TFs + REs
    TFRE_df = pd.DataFrame(np.zeros((len(TFRE_list),len(TFRE_list)),dtype=np.float32),index=TFRE_list,columns=TFRE_list)
    for item in TF_peak_union.values:
        TF = item[0]
        RE = item[2]
        if TF in TFs and RE in REs:
            TFRE_df.loc[TF,RE] = 1
            TFRE_df.loc[RE,TF] = 1
    TFRE_df.to_csv(adjdir / (gene_item + '_TFRE_adj.csv'))
    return TFRE_df     
def sc_nn_train(Target,inputs,adj_matrix,outdir,gene_id,gene_item):
    l1_lambda = 0.01
    activef='ReLU'
    alpha = 1
    eps=1e-12
    alpha = torch.tensor(alpha,dtype=torch.float32)
    targets = torch.tensor(Target[gene_id,:])
    inputs = torch.tensor(inputs,dtype=torch.float32)
    targets = targets.type(torch.float32)
    mean = inputs.mean(dim=1)
    std = inputs.std(dim=1)
    inputs = (inputs.T - mean) / (std+eps)
    inputs=inputs.T
    num_nodes=inputs.shape[0]
    y=targets.reshape(-1,1)     
    A = torch.tensor(adj_matrix, dtype=torch.float32)
    D = torch.diag(A.sum(1))
    degree = A.sum(dim=1)
    degree += eps
    D_sqrt_inv = 1 / degree.sqrt()
    D_sqrt_inv = torch.diag(D_sqrt_inv)
    L = D_sqrt_inv@(D - A)@D_sqrt_inv
    input_size = num_nodes
    mse_loss = nn.MSELoss()
    y_pred_all=0*(y+1-1)
    y_pred_all1=0*(y+1-1)
    y_pred_all1=y_pred_all1.numpy().reshape(-1)
    X_tr = inputs.T
    y_tr = y
    torch.manual_seed(seed_value)
    net = Net(input_size,activef)
    optimizer = Adam(net.parameters(),lr=0.01,weight_decay=l1_lambda)   
        #optimizer = Adam(net.parameters(),weight_decay=1)
        # Perform backpropagation
    Loss0=np.zeros([100,1])
    for i in range(100):
        # Perform forward pass
        y_pred = net(X_tr)
        # Calculate loss
        l1_norm = sum(torch.linalg.norm(p, 1) for p in net.parameters())
        lap_reg = alpha * torch.trace(torch.mm(torch.mm(net.fc1.weight, L), net.fc1.weight.t()))
        loss = mse_loss(y_pred, y_tr) +l1_norm*l1_lambda+lap_reg
        Loss0[i,0]=loss.detach().numpy()
        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    np.random.seed(42)
    background = X_tr[np.random.choice(X_tr.shape[0], 50, replace=True)]
    explainer = shap.DeepExplainer(net,background)
    shap_values = explainer.shap_values(X_tr, check_additivity=False)
    # torch.save(net,outdir+'net_'+gene_item+'.pt')
    # torch.save(shap_values,outdir+'shap_'+gene_item+'.pt')
    return shap_values


def sc_nn_train_perturb(Target,inputs,adj_matrix,outdir,gene_id,gene_item):
    l1_lambda = 0.01
    activef='ReLU'
    alpha = 1
    eps=1e-12
    alpha = torch.tensor(alpha,dtype=torch.float32)
    targets = torch.tensor(Target[gene_id,:])
    inputs = torch.tensor(inputs,dtype=torch.float32)
    targets = targets.type(torch.float32)
    mean = inputs.mean(dim=1)
    std = inputs.std(dim=1)
    inputs = (inputs.T - mean) / (std+eps)
    inputs=inputs.T
    num_nodes=inputs.shape[0]
    y=targets.reshape(-1,1)     
    A = torch.tensor(adj_matrix, dtype=torch.float32)
    D = torch.diag(A.sum(1))
    degree = A.sum(dim=1)
    degree += eps
    D_sqrt_inv = 1 / degree.sqrt()
    D_sqrt_inv = torch.diag(D_sqrt_inv)
    L = D_sqrt_inv@(D - A)@D_sqrt_inv
    input_size = num_nodes
    mse_loss = nn.MSELoss()
    y_pred_all=0*(y+1-1)
    y_pred_all1=0*(y+1-1)
    y_pred_all1=y_pred_all1.numpy().reshape(-1)
    X_tr = inputs.T
    y_tr = y
    torch.manual_seed(seed_value)
    net = Net(input_size,activef)
    optimizer = Adam(net.parameters(),lr=0.01,weight_decay=l1_lambda)   
        #optimizer = Adam(net.parameters(),weight_decay=1)
        # Perform backpropagation
    Loss0=np.zeros([100,1])
    for i in range(100):
        # Perform forward pass
        y_pred = net(X_tr)
        # Calculate loss
        l1_norm = sum(torch.linalg.norm(p, 1) for p in net.parameters())
        lap_reg = alpha * torch.trace(torch.mm(torch.mm(net.fc1.weight, L), net.fc1.weight.t()))
        loss = mse_loss(y_pred, y_tr) +l1_norm*l1_lambda+lap_reg
        Loss0[i,0]=loss.detach().numpy()
        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    np.random.seed(42)
    background = X_tr[np.random.choice(X_tr.shape[0], 50, replace=True)]
    explainer = shap.DeepExplainer(net,background)
    shap_values = explainer.shap_values(X_tr, check_additivity=False)
    # torch.save(net,outdir+'net_'+gene_item+'.pt')
    # torch.save(shap_values,outdir+'shap_'+gene_item+'.pt')
    return shap_values


def get_gene_LR_PriorityScore(shapvaluedir,shapdir,gene_LR_shapmat):
    target_id = list(gene_LR_shapmat.index)
    target_shap_list = []
    LR_list = list(gene_LR_shapmat.columns)
    tol = 1e-10
    lig_list = []
    rec_list = []
    for item in LR_list:
        temp = item.split(':')
        lig_list.append(temp[0])
        if '(' in temp[1]:
            r_temp = temp[1][1:-1]
            r_temp = r_temp.replace('+', '_')
            rec_list.append(r_temp)
        else:
            rec_list.append(temp[1])
    lig_list = list(set(lig_list))
    rec_list = list(set(rec_list))
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir + 'shap_value_' + gene_item + '.csv', index_col=0)
        shap_value_ligand_df = shap_value_df[lig_list]
        lig_mean_value = shap_value_ligand_df.mean().tolist()
        shap_value_receptor_df = shap_value_df[rec_list]
        rec_mean_value = shap_value_receptor_df.mean().tolist()
        lig_rank_df = pd.DataFrame(lig_mean_value, columns=['value'])
        lig_rank_df.index = lig_list
        lig_rank_df['rank'] = lig_rank_df['value'].rank(ascending=False)
        rec_rank_df = pd.DataFrame(rec_mean_value, columns=['value'])
        rec_rank_df.index = rec_list
        rec_rank_df['rank'] = rec_rank_df['value'].rank(ascending=False)
        LR_shape_list = []
        for LR_item in LR_list:
            if '(' in LR_item:
                ligand = LR_item.split(':')[0]
                receptor = LR_item.split(':')[1][1:-1]
                receptor = receptor.replace('+', '_')
                ligand_value = np.abs(shap_value_df[ligand].values).mean() + tol + 1/lig_rank_df.loc[ligand,'rank']
                receptor_value = np.abs(shap_value_df[receptor].values).mean() +tol + (1/rec_rank_df.loc[receptor,'rank']).mean()
                activity_score = ligand_value + receptor_value
            else:
                ligand = LR_item.split(':')[0]
                receptor = LR_item.split(':')[1]
                ligand_value = np.abs(shap_value_df[ligand].values).mean() + tol + 1/lig_rank_df.loc[ligand,'rank']
                receptor_value = np.abs(shap_value_df[receptor].values).mean() + tol + 1/rec_rank_df.loc[receptor,'rank']
                activity_score = ligand_value + receptor_value
            LR_shape_list.append(activity_score)
        target_shap_list.append(LR_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=LR_list)
    target_activity_inter.to_csv(shapdir + 'target_activity_PriorityScore_inter.csv')
    return target_activity_inter
def get_gene_ligand_shapmat_sc(shapvaluedir,shapdir,gene_ligand_shapmat,sample_labels,sc_type):
    target_id = list(gene_ligand_shapmat.index)
    target_shap_list = []
    L_list = list(gene_ligand_shapmat.columns)
    # tol = 1e-10
    # sample_labels = sample_labels.rename(columns={'label_atac': 'Label'})
    sample_labels = sample_labels.rename(columns={'cell_type': 'Label'})
    sample_labels['Sample Name'] = sample_labels.index
    sender_list = sample_labels[sample_labels['Label'] == sc_type]
    sender_list = list(sender_list.index)
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir + 'shap_value_' + gene_item + '.csv', index_col=0)
        L_shap_value_df = shap_value_df.loc[sender_list, :]
        L_shape_list = []
        for L_item in L_list:
            ligand = L_item
            ligand_value = np.abs(L_shap_value_df[ligand].values).mean()
            L_shape_list.append(ligand_value)
        target_shap_list.append(L_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=L_list)
    target_activity_inter.to_csv(shapdir + 'target_activity_ligand.csv')
    return target_activity_inter
def get_gene_receptor_shapmat_rc(shapvaluedir,shapdir,gene_receptor_shapmat,sample_labels,rc_type):
    target_id = list(gene_receptor_shapmat.index)
    target_shap_list = []
    R_list = list(gene_receptor_shapmat.columns)
    # tol = 1e-10
    # sample_labels = sample_labels.rename(columns={'label_atac': 'Label'})
    sample_labels = sample_labels.rename(columns={'cell_type': 'Label'})     
    sample_labels['Sample Name'] = sample_labels.index     
    receiver_list = sample_labels[sample_labels['Label'] == rc_type]     
    receiver_list = list(receiver_list.index)
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir + 'shap_value_' + gene_item + '.csv', index_col=0)
        R_shap_value_df = shap_value_df.loc[receiver_list, :]
        R_shape_list = []
        for R_item in R_list:
            receptor = R_item
            receptor_value = np.abs(R_shap_value_df[receptor].values).mean() 
            R_shape_list.append(receptor_value)
        target_shap_list.append(R_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=R_list)
    target_activity_inter.to_csv(shapdir + 'target_activity_receptor.csv')
    return target_activity_inter
def get_gene_ligand_shapmat(shapvaluedir,shapdir,gene_ligand_shapmat,database_filter1_mask):
    target_id = list(gene_ligand_shapmat.index)
    target_shap_list = []
    allL_list = list(gene_ligand_shapmat.columns)
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir / ('shap_value_LR_' + gene_item + '.csv'), index_col=0)
        sub_database_filter1 = database_filter1_mask[database_filter1_mask['TG_Symbol']==gene_item]
        geneL_list = sub_database_filter1['Ligand_Symbol'].tolist()
        L_shape_list = []
        for L_item in allL_list:
            if L_item in geneL_list:
                ligand = L_item
                ligand_value = np.abs(shap_value_df[ligand].values).mean()
                L_shape_list.append(ligand_value)
            else:
                L_shape_list.append(0)
        target_shap_list.append(L_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=allL_list)
    target_activity_inter.to_csv(shapdir / 'target_activity_ligand.csv')
    return target_activity_inter
def get_gene_receptor_shapmat(shapvaluedir,shapdir,gene_receptor_shapmat,database_filter1_mask):
    target_id = list(gene_receptor_shapmat.index)
    target_shap_list = []
    allR_list = list(gene_receptor_shapmat.columns)
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir / ('shap_value_LR_' + gene_item + '.csv'), index_col=0)
        sub_database_filter1 = database_filter1_mask[database_filter1_mask['TG_Symbol']==gene_item]
        geneR_list = sub_database_filter1['Receptor_Symbol'].tolist()
        R_shape_list = []
        for R_item in allR_list:
            if R_item in geneR_list:
                receptor = R_item
                receptor_value = np.abs(shap_value_df[receptor].values).mean()
                R_shape_list.append(receptor_value)
            else:
                R_shape_list.append(0)
        target_shap_list.append(R_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=allR_list)
    target_activity_inter.to_csv(shapdir / 'target_activity_receptor.csv')
    return target_activity_inter
def get_gene_RE_shapmat(shapvaluedir,shapdir,gene_RE_shapmat, adjdir):
    target_id = list(gene_RE_shapmat.index)
    target_shap_list = []
    allRE_list = list(gene_RE_shapmat.columns)
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir / ('shap_value_TFRE_' + gene_item + '.csv'), index_col=0)
        TFRE_adj = pd.read_csv(adjdir / (gene_item + '_TFRE_adj.csv'), index_col=0)
        TFRE_id = list(TFRE_adj.index)
        geneRE_list = [id for id in TFRE_id if id.startswith('chr')]
        RE_shape_list = []
        for RE_item in allRE_list:
            if RE_item in geneRE_list:
                TF = RE_item
                RE_value = np.abs(shap_value_df[TF].values).mean()
                RE_shape_list.append(RE_value)
            else:
                RE_shape_list.append(0)
        target_shap_list.append(RE_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=allRE_list)
    target_activity_inter.to_csv(shapdir / 'target_activity_RE.csv')
    return target_activity_inter
def get_gene_TFRE_shapmat(shapvaluedir,shapdir,gene_TFRE_shapmat):
    target_id = list(gene_TFRE_shapmat.index)
    target_shap_list = []
    TFRE_list = list(gene_TFRE_shapmat.columns)
    # tol = 1e-10
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir / ('shap_value_TFRE_' + gene_item + '.csv'))
        TFRE_shape_list = []
        for TFRE_item in TFRE_list:
            TF = TFRE_item.split(':')[0]
            RE = TFRE_item.split(':')[1]
            # RE = RE.replace('_','-')
            if TF in shap_value_df.columns:
                TF_value = np.abs(shap_value_df[TF].values).mean()
            else:
                TF_value = 0
            if RE in shap_value_df.columns:
                RE_value = np.abs(shap_value_df[RE].values).mean()
            else:
                RE_value = 0
            # activity_score = TF_value * RE_value
            activity_score = (TF_value + RE_value)/2
            TFRE_shape_list.append(activity_score)
        target_shap_list.append(TFRE_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=TFRE_list)
    target_activity_inter.to_csv(shapdir / 'target_activity_mean_intra.csv')
    return target_activity_inter
def get_gene_TFRE_PriorityScore(shapvaluedir,shapdir,gene_TFRE_shapmat,TF_len):
    target_id = list(gene_TFRE_shapmat.index)
    target_shap_list = []
    TFRE_list = list(gene_TFRE_shapmat.columns)
    tol = 1e-10
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir + 'shap_value_TFRE_' + gene_item + '.csv', index_col=0)
        shap_value_TF_df = shap_value_df.iloc[:,:TF_len]
        TF_mean_value = shap_value_TF_df.mean().tolist()
        shap_value_receptor_df = shap_value_df.iloc[:,TF_len:]
        RE_mean_value = shap_value_receptor_df.mean().tolist()
        TF_rank_df = pd.DataFrame(TF_mean_value, columns=['value'])
        TF_rank_df.index = shap_value_df.columns[:TF_len].tolist()
        TF_rank_df['rank'] = TF_rank_df['value'].rank(ascending=False)
        RE_rank_df = pd.DataFrame(RE_mean_value, columns=['value'])
        RE_rank_df.index = shap_value_df.columns[TF_len:].tolist()
        RE_rank_df['rank'] = RE_rank_df['value'].rank(ascending=False)
        TFRE_shape_list = []
        for TFRE_item in TFRE_list:
            TF = TFRE_item.split(':')[0]
            RE = TFRE_item.split(':')[1]
            RE = RE.replace('_','-')
            TF_value = np.abs(shap_value_df[TF].values).mean() + tol + 1/TF_rank_df.loc[TF,'rank']
            RE_value = np.abs(shap_value_df[RE].values).mean() + tol + 1/RE_rank_df.loc[RE,'rank']
            # activity_score = TF_value * RE_value
            activity_score = TF_value + RE_value
            TFRE_shape_list.append(activity_score)
        target_shap_list.append(TFRE_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=TFRE_list)
    target_activity_inter.to_csv(shapdir + 'target_activity_PriorityScore_intra.csv')
    return target_activity_inter
def get_gene_TF_shapmat(shapvaluedir,shapdir,gene_TF_shapmat, adjdir):
    target_id = list(gene_TF_shapmat.index)
    target_shap_list = []
    allTF_list = list(gene_TF_shapmat.columns)
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir / ('shap_value_TFRE_' + gene_item + '.csv'), index_col=0)
        TFRE_adj = pd.read_csv(adjdir / (gene_item + '_TFRE_adj.csv'), index_col=0)
        TFRE_id = list(TFRE_adj.index)
        geneTF_list = [id for id in TFRE_id if not id.startswith('chr')]
        TF_shape_list = []
        for TF_item in allTF_list:
            if TF_item in geneTF_list:
                TF = TF_item
                TF_value = np.abs(shap_value_df[TF].values).mean()
                TF_shape_list.append(TF_value)
            else:
                TF_shape_list.append(0)
        target_shap_list.append(TF_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=allTF_list)
    target_activity_inter.to_csv(shapdir / 'target_activity_TF.csv')
    return target_activity_inter
def get_gene_LR_shapmat(shapvaluedir,shapdir,gene_LR_shapmat):
    target_id = list(gene_LR_shapmat.index)
    target_shap_list = []
    LR_list = list(gene_LR_shapmat.columns)
    # tol = 1e-10
    for gene_item in tqdm(target_id):
        shap_value_df = pd.read_csv(shapvaluedir / ('shap_value_LR_' + gene_item + '.csv'))
        LR_shape_list = []
        for LR_item in LR_list:
            if '(' in LR_item:
                ligand = LR_item.split(':')[0]
                receptor = LR_item.split(':')[1][1:-1]
                receptor = receptor.replace('+', '_')
                if ligand in shap_value_df.columns:
                    ligand_value = np.abs(shap_value_df[ligand].values).mean()
                else:
                    ligand_value = 0
                selected_elements = [element for element in receptor if element in shap_value_df.columns]
                if not selected_elements:
                    receptor_value = 0
                else:
                    receptor_value = np.abs(shap_value_df[selected_elements].values).mean()
                # activity_score = ligand_value * receptor_value
                activity_score = (ligand_value + receptor_value)/2
            else:
                ligand = LR_item.split(':')[0]
                receptor = LR_item.split(':')[1]
                if ligand in shap_value_df.columns:
                    ligand_value = np.abs(shap_value_df[ligand].values).mean()
                else:
                    ligand_value = 0
                if receptor in shap_value_df.columns:
                    receptor_value = np.abs(shap_value_df[receptor].values).mean()
                # activity_score = ligand_value * receptor_value
                activity_score = (ligand_value + receptor_value)/2
            LR_shape_list.append(activity_score)
        target_shap_list.append(LR_shape_list)
    target_activity_inter = pd.DataFrame(target_shap_list, index=target_id, columns=LR_list)
    target_activity_inter.to_csv(shapdir / 'target_activity_LR_mean_inter.csv')
    return target_activity_inter