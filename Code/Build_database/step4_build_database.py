import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import polars as pl
from scipy.stats import zscore

import sys
sys.path.append('/home/nas2/biod/zhencaiwei/RegChat-main/Code/Build_database/LL_NET')  
import LL_net


def load_data(paths):
    LRTFDB_df = pd.read_csv(paths['LRTFDB'])
    TF_TG = json.load(open(paths['TF_TG']))
    Mm_expmat = pd.read_csv(paths['RNAmatrix'], index_col=0)
    TF_peak_dict = json.load(open(paths['TF_peak']))
    TG_peak = json.load(open(paths['TG_peak']))
    TF_peak_union = pd.read_csv(paths['TF_peak_union'])
    Mm_atacmat = pd.read_csv(paths['ATACmatrix'], index_col=0)
    return LRTFDB_df, TF_TG, Mm_expmat, TF_peak_dict, TG_peak, TF_peak_union, Mm_atacmat


def build_filter1(LRTFDB_df, TF_TG, Mm_expmat, outdir):
    lig, rec, tf, tg = [], [], [], []
    for ligand, receptor, tf_sym in tqdm(LRTFDB_df.values):
        if tf_sym in TF_TG:
            for tg_sym in TF_TG[tf_sym]:
                lig.append(ligand)
                rec.append(receptor)
                tf.append(tf_sym)
                tg.append(tg_sym)

    df = pd.DataFrame({
        'Ligand_Symbol': lig, 'Receptor_Symbol': rec,
        'TF_Symbol': tf, 'TG_Symbol': tg
    })
    
    mask_ligand = df['Ligand_Symbol'].isin(Mm_expmat.index)
    df = df[mask_ligand]
    split_receptor_symbols = df['Receptor_Symbol'].str.split('_')
    mask_receptor = split_receptor_symbols.apply(lambda symbols: all(symbol in Mm_expmat.index for symbol in symbols))
    df = df[mask_receptor]
    mask_TF = df['TF_Symbol'].isin(Mm_expmat.index)
    df = df[mask_TF]
    mask_tg = df['TG_Symbol'].isin(Mm_expmat.index)
    df = df[mask_tg]
    df_mask = df.reset_index(drop=True)

    df_mask.to_csv(outdir / 'database_filter1_mask.csv', index=False)
    return df_mask


def save_gene_idx(database, outdir):
    gene_idx_df = pd.DataFrame(database['TG_Symbol'].unique(), columns=['gene']).reset_index().rename(columns={'index': 'gene_idx'})
    gene_idx_df.to_csv(outdir / 'gene_idx.csv', index=False)
    return gene_idx_df


def build_inter_LR(database, Mm_expmat, gene_idx_df, adjdir, outdir, shapdir):
    # target_ids = gene_idx_df['gene'].tolist()
    target_ids = list(set(database['TG_Symbol'].values))

    for gene in tqdm(target_ids):
        shap_file = shapdir / f'shap_value_LR_{gene}.csv'
        if shap_file.exists():
            continue

        gene_idx = gene_idx_df.loc[gene_idx_df['gene'] == gene, 'gene_idx'].values
        LR_adj = LL_net.build_LRadj4gene(adjdir, database, gene)
        LR_adj_mat = pd.read_csv(adjdir / f"{gene}_LR_adj.csv", index_col=0).values

        sub_db = database[database['TG_Symbol'] == gene]
        lig_ids = sub_db['Ligand_Symbol'].unique().tolist()
        rec_ids = sub_db['Receptor_Symbol'].unique().tolist()

        L_exp = Mm_expmat.loc[lig_ids].values
        R_exp = []
        for r in rec_ids:
            if '_' in r:
                parts = r.split('_')
                vals = np.mean([Mm_expmat.loc[p].values for p in parts], axis=0)
                R_exp.append(vals)
            else:
                R_exp.append(Mm_expmat.loc[r].values)
        R_exp = np.vstack(R_exp)

        LR_exp = np.vstack([L_exp, R_exp])
        Target = Mm_expmat.loc[target_ids].values

        shap_val = LL_net.sc_nn_train(Target, LR_exp, LR_adj_mat, outdir, gene_idx, gene)
        shap_val_2d = np.squeeze(shap_val)
        lr_id = lig_ids + rec_ids
        pd.DataFrame(shap_val_2d, index=Mm_expmat.columns, columns=lr_id).to_csv(shap_file)


def RE_preprocess(database, TF_peak_dict, TG_peak, TF_peak_union, Mm_atacmat, outdir):
    REs_list = []
    for index, row in tqdm(database.iterrows()):
        TF_Symbol = row['TF_Symbol']
        TG_Symbol = row['TG_Symbol']
        
        if TF_Symbol not in TF_peak_dict:
            continue
        
        TF_peak_list = set(TF_peak_dict[TF_Symbol])
        TG_peak_list = set(TG_peak[TG_Symbol])
        common_value = TF_peak_list.intersection(TG_peak_list)
        REs_list += common_value
    REs_list = list(set(REs_list))
    RE_idx_df = pd.DataFrame(REs_list, columns=['RE'])
    RE_idx_df = RE_idx_df.reset_index().rename(columns={'index': 'RE_idx'})
    RE_idx_df.to_csv(outdir / 'RE_idx.csv', index=False)
    
    peak_names = list(Mm_atacmat.columns)
    peak_names_new = []
    for item in peak_names:
        item = item.replace(':','_')
        item = item.replace('-','_')
        peak_names_new.append(item)
    Mm_atacmat = Mm_atacmat.T
    Mm_atacmat.index = peak_names_new
    
    overlap_RE = []
    peaks = Mm_atacmat.index
    for item in RE_idx_df['RE']:
        if item in peaks:
            overlap_RE.append(item)
    overlap_RE_df = pd.DataFrame(overlap_RE, columns=['RE'])
    overlap_RE_df = overlap_RE_df.reset_index().rename(columns={'index': 'RE_idx'})
    overlap_RE_df.to_csv(outdir / 'overlap_RE_idx.csv', index=False)
    
    RE_list = overlap_RE_df['RE'].tolist()
    peak_rename_list = []
    for items in tqdm(TF_peak_union.values):
        TF = items[0]
        peak = items[1]
        if peak not in RE_list:
            for RE_item in RE_list:
                chrom = RE_item.split('_')[0]
                start = RE_item.split('_')[1]
                end = RE_item.split('_')[2]
                if ((peak.split('_')[0] == chrom and int(peak.split('_')[1]) <= int(start) and int(peak.split('_')[1]) >= int(end)) or
                    (peak.split('_')[0] == chrom and int(peak.split('_')[1]) >= int(start) and int(peak.split('_')[1]) <= int(end)) or
                    (peak.split('_')[0] == chrom and int(peak.split('_')[1]) <= int(end)) or
                    (peak.split('_')[0] == chrom and int(peak.split('_')[2]) >= int(start))):
                    peak_rename_list.append(RE_item)
                    break
                else:
                    peak_rename_list.append('xxx')
                    break
        else:
            peak_rename_list.append(peak)
    TF_peak_union['peak_rename'] = peak_rename_list
    TF_peak_union.to_csv(outdir / 'TF_peak_union_re.csv', index=False)
    return Mm_atacmat
    


def build_intra_TFRE(database, TF_peak_dict, TG_peak, TF_peak_union, Mm_expmat, Mm_atacmat, gene_idx_df, adjdir, outdir, shapdir):
    # target_ids = gene_idx_df['gene'].tolist()
    target_ids = list(set(database['TG_Symbol'].values))

    for gene in tqdm(target_ids):
        shap_file = shapdir / f'shap_value_TFRE_{gene}.csv'
        if shap_file.exists():
            continue

        gene_idx = gene_idx_df.loc[gene_idx_df['gene'] == gene, 'gene_idx'].values

        TFRE_adj = LL_net.build_TFREadj4gene(adjdir, database, TF_peak_union, gene, TG_peak, Mm_expmat, Mm_atacmat)
        TFRE_adj_mat = pd.read_csv(adjdir / f"{gene}_TFRE_adj.csv", index_col=0)
        TFRE_id = TFRE_adj_mat.index.tolist()
        TF_id = [x for x in TFRE_id if not x.startswith('chr')]
        RE_id = [x for x in TFRE_id if x.startswith('chr')]

        TF_exp = Mm_expmat.loc[TF_id]
        RE_exp = Mm_atacmat.loc[RE_id]

        TFRE_exp = pd.concat([TF_exp, RE_exp], axis=0).values
        Target = Mm_expmat.loc[target_ids].values

        shap_val = LL_net.sc_nn_train(Target, TFRE_exp, TFRE_adj_mat.values, outdir, gene_idx, gene)
        shap_val_2d = np.squeeze(shap_val)
        pd.DataFrame(shap_val_2d, index=Mm_expmat.columns, columns=TFRE_id).to_csv(shap_file)





def process_LR_pairs(database_df):
    LRpairDB = database_df[['Ligand_Symbol', 'Receptor_Symbol']]
    pair_list = []
    
    for ligand, receptor in LRpairDB.values:
        if '_' in receptor:
            receptor = receptor.replace('_', '+')
            pair = f"{ligand}:({receptor})"
        else:
            pair = f"{ligand}:{receptor}"
        pair_list.append(pair)
    
    LRpairDB_merge = pd.DataFrame({'pair': pair_list}).drop_duplicates().reset_index(drop=True)
    return LRpairDB_merge


def extract_ligands_receptors(LRpairDB_merge):
    lig_list, rec_list = [], []
    
    for item in LRpairDB_merge['pair']:
        lig, rec = item.split(':')
        lig_list.append(lig)
        
        if '(' in rec:
            rec = rec[1:-1].replace('+', '_')
        rec_list.append(rec)
    
    return list(set(lig_list)), list(set(rec_list))


def filter_by_shap_zscore_inter(shap_mat, database_df, zscore_threshold=2):
    nozero_shapmat = shap_mat.loc[:, ~(shap_mat == 0).all()]
    
    zscore_mat = nozero_shapmat.apply(zscore, axis=1)
    mask = zscore_mat.applymap(lambda x: abs(x) >= zscore_threshold)
    selected = zscore_mat.where(mask)
    
    zscore_dict = {
        index: selected.columns[row.dropna().values.nonzero()[0]].tolist() 
        for index, row in selected.iterrows() 
        if row.dropna().any()
    }
    
    filtered_data = []
    for lig, rec, tf, tg in database_df.values:
        if tg in zscore_dict:
            features = zscore_dict[tg]
            if (lig in features) or (rec in features) or (f"{lig}:{rec}" in features) or (f"{lig}:({rec.replace('_','+')})" in features) :
                filtered_data.append([lig, rec, tf, tg])
    
    return pd.DataFrame(filtered_data, columns=database_df.columns)


def filter_by_shap_zscore_intra(shap_mat, database_df, TF_peak_dict, zscore_threshold=2, mode='TF'):
        nozero_shapmat = shap_mat.loc[:, ~(shap_mat == 0).all()]
    
        zscore_mat = nozero_shapmat.apply(zscore, axis=1)
        mask = zscore_mat.applymap(lambda x: abs(x) >= zscore_threshold)
        selected = zscore_mat.where(mask)
        
        zscore_dict = {
            index: selected.columns[row.dropna().values.nonzero()[0]].tolist()
            for index, row in selected.iterrows()
            if row.dropna().any()
        }
        
        filtered_data = []
        for lig, rec, tf, tg in database_df.values:
            if tg in zscore_dict:
                features = zscore_dict[tg]
                
                if mode == 'TF':
                    if tf in features:
                        filtered_data.append([lig, rec, tf, tg])
                elif mode == 'RE':
                    if tf in TF_peak_dict:  
                        tg_peaks = set(features)
                        tf_peaks = set(TF_peak_dict[tf])
                        if tg_peaks & tf_peaks:
                            filtered_data.append([lig, rec, tf, tg])
                elif mode == 'TFRE':
                    for tfre in features:
                        tf_part = tfre.split(':')[0]
                        if tf == tf_part:
                            filtered_data.append([lig, rec, tf, tg])
                            break
        
        return pd.DataFrame(filtered_data, columns=database_df.columns)
    


def filter_interactions(database_df, overlap_RE_df,TF_peak_union, TF_peak_map_dict, shapvalue_dir, shapdir, adjdir=None, mode='ligand'):
    LRpairDB_merge = process_LR_pairs(database_df)
    lig_list, rec_list = extract_ligands_receptors(LRpairDB_merge)
    target_ids = database_df['TG_Symbol'].unique()
    
    if mode == 'ligand':
        shap_mat = pd.DataFrame(np.zeros((len(target_ids), len(lig_list)), dtype=np.float32), 
                               index=target_ids, columns=lig_list)
        shap_mat = LL_net.get_gene_ligand_shapmat(shapvalue_dir, shapdir, shap_mat, database_df)
        return filter_by_shap_zscore_inter(shap_mat, database_df)
    elif mode == 'receptor':
        shap_mat = pd.DataFrame(np.zeros((len(target_ids), len(rec_list)), dtype=np.float32),
                               index=target_ids, columns=rec_list)
        shap_mat = LL_net.get_gene_receptor_shapmat(shapvalue_dir, shapdir, shap_mat, database_df)
        return filter_by_shap_zscore_inter(shap_mat, database_df)
    elif mode == 'LR':
        shap_mat = pd.DataFrame(np.zeros((len(target_ids), len(LRpairDB_merge)), dtype=np.float32),
                               index=target_ids, columns=LRpairDB_merge['pair'].values)
        shap_mat = LL_net.get_gene_LR_shapmat(shapvalue_dir, shapdir, shap_mat)
        return filter_by_shap_zscore_inter(shap_mat, database_df)
    elif mode == 'TF':
        tf_list = database_df['TF_Symbol'].unique().tolist()
        shap_mat = pd.DataFrame(np.zeros((len(target_ids), len(tf_list)), dtype=np.float32),
                               index=target_ids, columns=tf_list)
        shap_mat = LL_net.get_gene_TF_shapmat(shapvalue_dir, shapdir, shap_mat, adjdir)
        return filter_by_shap_zscore_intra(shap_mat, database_df,TF_peak_map_dict, zscore_threshold=2, mode='TF')
    elif mode == 'RE':
        re_list = list(overlap_RE_df['RE'].tolist())
        shap_mat = pd.DataFrame(np.zeros((len(target_ids), len(re_list)), dtype=np.float32),
                               index=target_ids, columns=re_list)
        shap_mat = LL_net.get_gene_RE_shapmat(shapvalue_dir, shapdir, shap_mat, adjdir)
        return filter_by_shap_zscore_intra(shap_mat, database_df,TF_peak_map_dict, zscore_threshold=2, mode='RE')
    elif mode == 'TFRE':
        tfre_pairs = []
        for tf, _, re in TF_peak_union.values:
            if re != 'xxx':
                tfre_pairs.append(f"{tf}:{re}")
        tfre_pairs = list(set(tfre_pairs))
        shap_mat = pd.DataFrame(np.zeros((len(target_ids), len(tfre_pairs)), dtype=np.float32),
                               index=target_ids, columns=tfre_pairs)
        shap_mat = LL_net.get_gene_TFRE_shapmat(shapvalue_dir, shapdir, shap_mat)
        return filter_by_shap_zscore_intra(shap_mat, database_df,TF_peak_map_dict, zscore_threshold=2, mode='TFRE')
    



def build_final_database(inter_files, intra_files, output_path):
    inter_dfs = [pl.read_csv(f).to_pandas() for f in inter_files]
    union_inter = pd.concat(inter_dfs).drop_duplicates().reset_index(drop=True)
    
    intra_dfs = [pl.read_csv(f).to_pandas() for f in intra_files]
    union_intra = pd.concat(intra_dfs).drop_duplicates().reset_index(drop=True)

    final_db = pd.merge(union_intra, union_inter, how='inner').reset_index(drop=True)
    final_db.to_csv(output_path, index=False)
    return final_db
    


def main():
    config = {
        'paths': {
            'LRTFDB': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/complexLRTFDB.csv',
            'TF_TG': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output/TF_TG.json',
            'RNAmatrix': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/RNAmatrix.csv',
            'TF_peak': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output/TF_peak_JASPAR_ArchR.json',
            'TG_peak': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output/TG_peak.json',
            'TF_peak_union': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output/TF_peak_union.csv',
            'ATACmatrix': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/ATACmatrix.csv'
        },
        'outdir': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output',
        'adjdir': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output/adj4gene',
        'shapdir': '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output/values',
        "dbdir": '/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output/database'
    }

    outdir = Path(config['outdir']); outdir.mkdir(exist_ok=True, parents=True)
    adjdir = Path(config['adjdir']); adjdir.mkdir(exist_ok=True, parents=True)
    shapdir = Path(config['shapdir']); shapdir.mkdir(exist_ok=True, parents=True)
    dbdir = Path(config['dbdir']); dbdir.mkdir(exist_ok=True, parents=True)
    
    print("Loading data...")
    LRTFDB_df, TF_TG, Mm_expmat, TF_peak_dict, TG_peak, TF_peak_union, Mm_atacmat = load_data(config['paths'])
    print("Data loading done. ")
    
    print("Filtering database...")
    filter1_df = build_filter1(LRTFDB_df, TF_TG, Mm_expmat, outdir)
    print("Database filtering done.")
    
    print("Extracting gene index...")
    gene_idx_df = save_gene_idx(filter1_df, outdir)
    print("Gene index extraction done.")

    # build inter (LR–TG)
    print("Building inter LR for each TG...")
    build_inter_LR(filter1_df, Mm_expmat, gene_idx_df, adjdir, outdir, shapdir)
    print("Inter LR building done.")
    
    print("Preprocessing REs befor intra...")
    Mm_atacmat_new = RE_preprocess(filter1_df, TF_peak_dict, TG_peak, TF_peak_union, Mm_atacmat, outdir)
    TF_peak_union_re = pd.read_csv(outdir / 'TF_peak_union_re.csv', sep=',')
    print("REs preprocessing done.")

    # build intra (TF/RE–TG)
    print("Building intra TFRE for each TG...")
    build_intra_TFRE(filter1_df, TF_peak_dict, TG_peak, TF_peak_union_re, Mm_expmat, Mm_atacmat_new, gene_idx_df, adjdir, outdir, shapdir)
    print("Intra TFRE building done.")
    
    print("Filtering ligand interactions...")
    # data for accelerated filtering     
    overlap_RE_df = pd.read_csv(outdir / 'overlap_RE_idx.csv', index_col=0)
    TF_peak_map_dict = json.load(open(outdir / 'tf_peak_mapping_dict.json'))
    ligand_filtered = filter_interactions(filter1_df,overlap_RE_df,TF_peak_union_re, TF_peak_map_dict, shapdir, outdir, mode='ligand')
    ligand_filtered.to_csv(dbdir / 'database_filter2_ligand2_inter.csv', index=False)
    
    print("Filtering receptor interactions...")
    receptor_filtered = filter_interactions(filter1_df, overlap_RE_df, TF_peak_union_re, TF_peak_map_dict, shapdir, outdir, mode='receptor')
    receptor_filtered.to_csv(dbdir / 'database_filter2_receptor2_inter.csv', index=False)
    
    print("Filtering LR pairs...")
    lr_filtered = filter_interactions(filter1_df, overlap_RE_df, TF_peak_union_re, TF_peak_map_dict, shapdir, outdir, mode='LR')
    lr_filtered.to_csv(dbdir / 'database_filter2_mean2_inter.csv', index=False)
    
    print("Filtering TF interactions...")
    tf_filtered = filter_interactions(filter1_df, overlap_RE_df, TF_peak_union_re, TF_peak_map_dict, shapdir, outdir, adjdir=adjdir, mode='TF')
    tf_filtered.to_csv(dbdir / 'database_filter2_TF2_intra.csv', index=False)
    
    print("Filtering RE interactions...")
    re_filtered = filter_interactions(filter1_df, overlap_RE_df, TF_peak_union_re, TF_peak_map_dict, shapdir, outdir, adjdir=adjdir, mode='RE')
    re_filtered.to_csv(dbdir / 'database_filter2_RE2_intra.csv', index=False)
    
    print("Filtering TFRE pairs...")
    tfre_filtered = filter_interactions(filter1_df, overlap_RE_df, TF_peak_union_re, TF_peak_map_dict, shapdir, outdir, mode='TFRE')
    tfre_filtered.to_csv(dbdir / 'database_filter2_mean2_intra.csv', index=False)
    
    print("Building final database...")
    inter_files = [
        dbdir / 'database_filter2_ligand2_inter.csv',
        dbdir / 'database_filter2_receptor2_inter.csv',
        dbdir / 'database_filter2_mean2_inter.csv'
    ]
    
    intra_files = [
        dbdir / 'database_filter2_TF2_intra.csv',
        dbdir / 'database_filter2_RE2_intra.csv',
        dbdir / 'database_filter2_mean2_intra.csv'
    ]
    
    final_db = build_final_database(inter_files, intra_files, dbdir / 'intersection_all_DB.csv')
    print("Database construction completed.")
    print("Final database shape:", final_db.shape)
    print("Final database saved to:", dbdir / 'intersection_all_DB.csv')
    
    


if __name__ == "__main__":
    main()
