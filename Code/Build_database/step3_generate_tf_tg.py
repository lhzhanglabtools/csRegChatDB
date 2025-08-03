import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
import scanpy as sc
from pathlib import Path
from collections import defaultdict


def load_data(config):
    paths = config
    LRTFDB_df = pd.read_csv(paths['LRTFDB'])
    peak_TF = pd.read_csv(paths['peak_TF'], sep='\t')
    ligand_receptor_tf = pd.read_csv(paths['ligand_receptor_tf'], sep='\t')
    ArchR_TF_peak = pd.read_csv(paths['ArchR_TF_peak'])
    adata_rna = sc.read(paths['adata_rna'])
    cell_type = pd.read_csv(paths['cell_type'], index_col=0)
    mm10_genes_df = pd.read_csv(paths['mm10_genes'], sep='\t', names=['name', 'chrom', 'strand', 'txStart', 'txEnd', 'name2'])
    mm10_chrom_sizes = pd.read_csv(paths['mm10_chrom_sizes'], sep='\t', names=['chrom', 'size'])
    return LRTFDB_df, peak_TF, ligand_receptor_tf, ArchR_TF_peak, adata_rna, cell_type, mm10_genes_df, mm10_chrom_sizes


def extract_TF_lists(LRTFDB_df, peak_TF, ligand_receptor_tf, ArchR_TF_peak):
    cellcall_TF_list = ligand_receptor_tf['TF_Symbol'].tolist()
    TF_list_JASPAR = peak_TF['TF'].tolist()
    TF_list_call = LRTFDB_df['TF_Symbol'].tolist()
    ArchR_TF_list = [item.split('_')[0] for item in ArchR_TF_peak['TFs'].tolist()]

    call_JASPAR_TF_intersect = set(TF_list_JASPAR) & set(TF_list_call)
    call_ArchR_TF_intersect = set(ArchR_TF_list) & set(TF_list_call)
    call_TF_union = list(call_JASPAR_TF_intersect | call_ArchR_TF_intersect)
    return call_JASPAR_TF_intersect, call_ArchR_TF_intersect, call_TF_union


def build_TF_peak_dict(peak_TF, ArchR_TF_peak, call_JASPAR_TF_intersect, call_ArchR_TF_intersect):
    TF_peak_dict = defaultdict(list)

    for _, row in peak_TF[peak_TF['TF'].isin(call_JASPAR_TF_intersect)].iterrows():
        TF_peak_dict[row['TF']].append(row['loci'])

    for _, row in ArchR_TF_peak[ArchR_TF_peak['TFs'].isin(call_ArchR_TF_intersect)].iterrows():
        TF_peak_dict[row['TFs']].append(row['peaks'])
    
    with open('data/TF_peak_JASPAR_ArchR.json', 'w') as f:
        json.dump(TF_peak_dict, f)

    return TF_peak_dict


def save_TF_peak_union(peak_TF, ArchR_TF_peak, call_JASPAR_TF_intersect, call_ArchR_TF_intersect, out_path):
    TF_peak_JASPAR = peak_TF[peak_TF['TF'].isin(call_JASPAR_TF_intersect)]
    TF_peak_ArchR = ArchR_TF_peak[ArchR_TF_peak['TFs'].isin(call_ArchR_TF_intersect)]

    TF_peak_union = pd.concat([
        TF_peak_JASPAR.rename(columns={'loci': 'peak'})[['TF', 'peak']],
        TF_peak_ArchR.rename(columns={'TFs': 'TF', 'peaks': 'peak'})[['TF', 'peak']]
    ]).drop_duplicates().reset_index(drop=True)

    TF_peak_union.to_csv(out_path, index=False)
    return TF_peak_union


def extract_marker_regions(adata_rna, cell_type, mm10_genes_df, mm10_chrom_sizes, out_path):
    adata_rna.obs['cell_type'] = cell_type['cell_type']
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.highly_variable_genes(adata_rna, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.tl.rank_genes_groups(adata_rna, 'cell_type', method='t-test')

    marker_genes = pd.concat([
        pd.DataFrame({
            'gene': adata_rna.uns['rank_genes_groups']['names'][group][:50],
            'cluster': group
        })
        for group in adata_rna.uns['rank_genes_groups']['names'].dtype.names
    ], ignore_index=True)

    marker_genes.to_csv(out_path / 'marker_genes.csv', index=False)

    marker_genes_unique = marker_genes['gene'].unique()
    common_names = mm10_genes_df[mm10_genes_df['name2'].isin(marker_genes_unique)]

    marker_region = []
    for _, row in common_names.iterrows():
        chrom, tss = row['chrom'], int(row['txStart'])
        chrom_size = int(mm10_chrom_sizes.loc[mm10_chrom_sizes['chrom'] == chrom, 'size'].iloc[0])
        start = max(0, tss - 250_000)
        end = min(chrom_size, tss + 250_000)
        marker_region.append({'gene': row['name2'], 'chrom': chrom, 'start': start, 'end': end})

    marker_region_df = pd.DataFrame(marker_region).set_index('gene')
    marker_region_df.to_csv(out_path / 'marker_region_250kb.csv')
    return marker_region_df


def build_TG_peak_dict(TF_peak_union, marker_region, out_path):
    TG_peak = defaultdict(list)

    peaks_df = TF_peak_union['peak'].str.split('_', expand=True)
    peaks_df.columns = ['chrom', 'start', 'end']
    peaks_df['start'] = peaks_df['start'].astype(int)
    peaks_df['end'] = peaks_df['end'].astype(int)

    for gene, reg in tqdm(marker_region.iterrows(), total=len(marker_region)):
        overlap = peaks_df[
            (peaks_df['chrom'] == reg['chrom']) &
            (peaks_df['start'] >= reg['start']) &
            (peaks_df['end'] <= reg['end'])
        ]
        for _, peak_row in overlap.iterrows():
            peak_str = f"{peak_row['chrom']}_{peak_row['start']}_{peak_row['end']}"
            TG_peak[gene].append(peak_str)

    with open(out_path / 'TG_peak.json', 'w') as f:
        json.dump(TG_peak, f)

    return TG_peak


def build_TF_TG_dict(TF_peak_dict, TG_peak, out_path):
    TF_TG = defaultdict(list)
    for tf, tf_peaks in tqdm(TF_peak_dict.items()):
        for tg, tg_peaks in TG_peak.items():
            if set(tf_peaks) & set(tg_peaks):
                if tf_peaks in TF_TG:
                    TF_TG[tf].append(tg)
                else:
                    TF_TG[tf] = [tg]

    with open(out_path / 'TF_TG.json', 'w') as f:
        json.dump(TF_TG, f)

    return TF_TG


def main():
    config = {
        "output_dir": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/output",
        "paths": {
            "LRTFDB": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/complexLRTFDB.csv",
            "peak_TF": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/MIASR_peaks_used_TFs_JASPAR2024.txt",
            "ligand_receptor_tf": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/new_ligand_receptor_TFs_homology.txt",
            "ArchR_TF_peak": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/ArchR_peaks_used_TFs_1.csv",
            "adata_rna": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/adata_rna.h5ad",
            "cell_type": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/cell_type.csv",
            "mm10_genes": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/mm10.gene.txt",
            "mm10_chrom_sizes": "/home/nas2/biod/zhencaiwei/RegChat-main/RegChat_Real_Datasets/MISAR/mm10.chrom.sizes.txt"
        }
    }

    out_path = Path(config['output_dir'])
    out_path.mkdir(exist_ok=True, parents=True)
    
    print("Loading data...")
    LRTFDB_df, peak_TF, ligand_receptor_tf, ArchR_TF_peak, adata_rna, cell_type, mm10_genes_df, mm10_chrom_sizes = load_data(config['paths'])
    print("Data loading done.")
    
    print("Extracting TF lists...")
    call_JASPAR_TF, call_ArchR_TF, _ = extract_TF_lists(LRTFDB_df, peak_TF, ligand_receptor_tf, ArchR_TF_peak)
    print("TF lists extraction done.")
    
    print("Building TF peak dictionary...")
    TF_peak_dict = build_TF_peak_dict(peak_TF, ArchR_TF_peak, call_JASPAR_TF, call_ArchR_TF)
    print("TF peak dictionary built.")
    
    print("Saving TF peak union...")
    TF_peak_union = save_TF_peak_union(peak_TF, ArchR_TF_peak, call_JASPAR_TF, call_ArchR_TF, out_path / 'TF_peak_union.csv')
    print("TF peak union saved.")
    
    print("Extracting marker regions...")
    marker_region = extract_marker_regions(adata_rna, cell_type, mm10_genes_df, mm10_chrom_sizes, out_path)
    print("Marker regions extracted and saved.")
    
    print("Building TG peak dictionary...")
    TG_peak = build_TG_peak_dict(TF_peak_union, marker_region, out_path)
    print("TG peak dictionary built.")
    
    print("Building TF-TG dictionary...")
    build_TF_TG_dict(TF_peak_dict, TG_peak, out_path)
    print("TF-TG dictionary built and saved.")


if __name__ == "__main__":
    main()
