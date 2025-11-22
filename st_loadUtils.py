# Created by Zhoufanghui at 2025/4/13
# 读取数据集，添加注释
import os

import pandas as pd
import scanpy as sc
import squidpy as sq


# -------------- 10X Genomics Visium ---------------------
# D:\WorkSpace\Datasets\6-DLPFC
# 除掉NA值
def load_DLPFC(root_dir='D:/WorkSpace/Datasets/6-DLPFC', section_id='151672'):
    ad = sc.read_visium(path=os.path.join(root_dir, section_id), count_file='filtered_feature_bc_matrix.h5')
    ad.var_names_make_unique()

    Ann_df = pd.read_csv(os.path.join(root_dir, section_id, section_id + '_truth.txt'), sep='\t', header=None,
                         index_col=0)
    Ann_df.columns = ['Ground Truth']
    ad.obs['original_clusters'] = Ann_df.loc[ad.obs_names, 'Ground Truth']

    keep_bcs = ad.obs.dropna().index
    ad = ad[keep_bcs].copy()
    return ad

# D:\WorkSpace\Datasets\11-Mouse Brain (sagittal)\mMAMP\MA
# cluster=52
def load_mMAMP(root_dir='D:/WorkSpace/Datasets/11-Mouse Brain (sagittal)/mMAMP', section_id='MA'):
    ad = sc.read_visium(path=os.path.join(root_dir, section_id),
                        count_file=section_id + '_filtered_feature_bc_matrix.h5')
    ad.var_names_make_unique()

    gt_dir = os.path.join(root_dir, section_id, 'gt')
    gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep='\t', header=0, index_col=0)
    ad.obs = gt_df
    ad.obs['original_clusters'] = ad.obs['ground_truth']
    return ad

# D:\WorkSpace\Datasets\13-HBCA1\V1_Human_Breast_Cancer_Block_A_Section_1
# cluster=20    radios=300可以测出来5-7个邻居
def load_HBCA1(root_dir='D:/WorkSpace/Datasets/13-HBCA1', section_id='V1_Human_Breast_Cancer_Block_A_Section_1'):
    ad = sc.read_visium(path=os.path.join(root_dir, section_id), count_file='filtered_feature_bc_matrix.h5')
    ad.var_names_make_unique()

    gt_dir = os.path.join(root_dir, section_id, 'gt')
    gt_df = pd.read_csv(os.path.join(gt_dir, 'tissue_positions_list_GTs.txt'), sep='\t', header=0, index_col=0)
    ad.obs['original_clusters'] = gt_df.loc[ad.obs_names, 'fine_annot_type']
    return ad

# squidpy包里内置的数据集，小鼠大脑，自带注释cluster
# D:\WorkSpace\Datasets\12-Mouse Brain (Coronal)
# cluster=15
def load_mBrain():
    # img = sq.datasets.visium_hne_image()
    ad = sq.datasets.visium_hne_adata()
    ad.obs['original_clusters'] = ad.obs['cluster']
    return ad

# ----------------- STARmap --------------------------
# D:\WorkSpace\Datasets\16-MVC\STARmap_mouse_visual_cortex
# cluster=7
def load_mvc(root_dir='D:/WorkSpace/Datasets/16-MVC/STARmap_mouse_visual_cortex',
             section_id='STARmap_20180505_BY3_1k.h5ad'):
    ad = sc.read(os.path.join(root_dir, section_id))
    ad.var_names_make_unique()
    ad.obs['original_clusters'] = ad.obs['label']
    # sc.pl.embedding(ad, basis="spatial", color="original_clusters")
    return ad

# D:/WorkSpace/Datasets/2-Mouse Brain (medial prefrontal cortex)
# section id = '20180417_BZ5_control', '20180419_BZ9_control', '20180424_BZ14_control' 3 in total
# cluster       4                       4                       4
def load_mPFC(root_dir='D:/WorkSpace/Datasets/2-Mouse Brain (medial prefrontal cortex)',
             section_id='20180417_BZ5_control'):
    coord = pd.read_csv(os.path.join(root_dir,'Mouse Brain coord',section_id+'_coord.csv'), index_col=0, header=0)
    coord.columns = ['x', 'y', 'c', 'z']
    ad = sc.read_csv(os.path.join(root_dir, 'Mouse Brain data', section_id+'_counts.csv')).transpose()
    ad.obs = coord
    ad.obsm["spatial"] = coord.loc[:, ['x', 'y']].to_numpy()
    ad.obs['Cell class'] = coord.loc[:, 'c'].values
    ad.obs['original_clusters'] = coord.loc[:, 'z'].values.astype(int).astype(str)

    # sc.pl.embedding(adata, basis="spatial", color="original_clusters")
    return ad

# --------------------- MERFISH -----------------------
# D:\WorkSpace\Datasets\1-Mouse Hypothalamus
# 连续里有z作为类别，其他里没有
# section id = '0.26', '0.21', '0.16', '0.11', '0.06', '0.01', '-0.04', '-0.09', '-0.14', '-0.19', '-0.24', '-0.29' 12 in total
# cluster =     15      15      14      15      15      15      14       15       15       15      16        15
def load_mHypothalamus(root_dir='D:/WorkSpace/Datasets/1-Mouse Hypothalamus', section_id='-0.04'):

    # 加一个if部分
    coord = pd.read_csv(os.path.join(root_dir, 'Mouse Hypothalamus coord','连续大脑下丘脑区域', section_id + '_coord.csv'), index_col=0, header=0)
    ad = sc.read_csv(os.path.join(root_dir, 'Mouse Hypothalamus data', '连续大脑下丘脑区域',section_id + '_counts.csv')).transpose()
    ad.obs = coord
    ad.obsm["spatial"] = coord.loc[:, ['x', 'y']].to_numpy()
    # ad.obs['Neuron cluster ID'] = coord.loc[:, 'Neuron_cluster_ID'].values
    # ad.obs['Cell_class'] = coord.loc[:, 'Cell_class'].values
    ad.obs['original_clusters'] = coord.loc[:, 'z'].values

    # sc.pl.embedding(ad, basis="spatial", color="original_clusters")
    return ad

# ----------------------- ST ----------------------------
# D:\WorkSpace\Datasets\10-Human Breast
# ########## 预处理？
# section id = A1(348) B1(295) C1(177) D1(309) E1(587) F1(692) G2(475) H1(613) ~J1(254), 8 in total
# clusters =   6       5       4       4       4       4       7       7
def load_HER2_tumor(root_dir='D:/WorkSpace/Datasets/10-Human Breast/H5adData', section_id='A1'):
    ad = sc.read_h5ad(os.path.join(root_dir, section_id+'.h5ad'))
    ad.obs['original_clusters'] = ad.obs['label']
    # sc.pl.embedding(ad, basis="spatial", color="original_clusters")
    return ad

# 小鼠嗅球数据   9
# 没有注释 没有预处理
# files_1 = ["Layer1_BC", "Layer2_BC", "Layer3_BC", "Layer4_BC"]
# files_2 = ["Rep1", "Rep2", "Rep3", "Rep4"]
# files_3 = ["Rep5_MOB", "Rep6_MOB", "Rep7_MOB", "Rep8_MOB",
#            "Rep9_MOB", "Rep10_MOB", "Rep11_MOB", "Rep12_MOB"]

def load_st_olf(root_dir='D:/WorkSpace/Datasets/9-Mouse Brain (olfactory bulb)/H5adData', section_id='Layer1_BC'):
    ad = sc.read_h5ad(os.path.join(root_dir, section_id+'.h5ad'))
    return ad


# -------------------- Stereo-seq ------------------------
# 4 D:\WorkSpace\Datasets\4-Mouse Brain (olfactory Bulb)
# 没有预处理
# "Mouse_olfa_S1", "Mouse_olfa_S2"
# cluster=12
def load_stereo_olf(root_dir='D:/WorkSpace/Datasets/4-Mouse Brain (olfactory Bulb)', section_id='Mouse_olfa_S1'):
    ad = sc.read_h5ad(os.path.join(root_dir, section_id+'.h5ad'))
    ad.obs['original_clusters'] = ad.obs['annotation']
    return ad

# 胚胎 17
def load_embyo(root_dir='D:/WorkSpace/Datasets/17-Membyo', section_id='E9.5_E1S1.MOSTA'):
    ad = sc.read_h5ad(os.path.join(root_dir, section_id+'.h5ad'))
    ad.obs['original_clusters'] = ad.obs['annotation']
    return ad


# -------------------- Slide-seq ---------------------------
# squidpy包里内置的数据集，自带注释cluster  对应14
def load_SlideseqV2():
    ad = sq.datasets.slideseqv2()
    ad.obs['original_clusters'] = ad.obs['cluster']
    # sq.pl.spatial_scatter(adata, color="cluster", size=1, shape=None)
    return ad

# ------------------- osmFISH -------------------------------

# 3 D:\WorkSpace\Datasets\3-Mouse Brain (somatosensory cortex)
# 要安装loompy
def load_osmFISH(root_dir='D:/WorkSpace/Datasets/3-Mouse Brain (somatosensory cortex)',
                 section_id='osmFISH_SScortex_mouse_all_cells.loom'):
    ad = sc.read_loom(os.path.join(root_dir, section_id), sparse=True)
    ad.obsm["spatial"] = ad.obs.loc[:, ['X', 'Y']].to_numpy()
    ad = ad[ad.obs['Valid'] == 1, :]
    ad.obs['original_clusters'] = ad.obs['Region']
    return ad
