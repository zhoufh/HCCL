# Created by Zhoufanghui at 2025/7/22
import os
import random

import numpy as np
import ot
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn.neighbors
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import homogeneity_score, completeness_score, calinski_harabasz_score, silhouette_score, v_measure_score
from sklearn.metrics import normalized_mutual_info_score as NMI


# 预处理
def preprocess(adata, top_genes=3000):
    if adata.X.shape[1] < top_genes:
        genes = adata.X.shape[1]
        n_genes = min(600, genes * 0.6)
    else:
        genes = top_genes
        n_genes = genes * 0.2
    sc.pp.filter_genes(adata, min_cells=3)  # 过滤低表达的基因
    id_tmp = np.asarray([not str(name).startswith("ERCC") for name in adata.var_names], dtype=bool)
    adata._inplace_subset_var(id_tmp)  # 过滤ERCC人工合成的外源RNA
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=int(n_genes))  # 基于表达的均值进行筛选 600
    adata.var['top_highly_variable'] = adata.var['highly_variable']
    # print(adata.var['top_highly_variable'].sum())
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=genes)  # 基于表达的均值进行筛选
    sc.pp.normalize_total(adata)  # 归一化，消除测序差异
    sc.pp.log1p(adata)  # 对数变换
    sc.pp.scale(adata, zero_center=False, max_value=10)  # 标准化，使不同基因表达量在相同范围
    adata = adata[:, adata.var['highly_variable'] == True]
    return adata


# 固定全局seed，使可复现
def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 设置运行环境
    torch.backends.cudnn.deterministic = True  # 在卷积时模型的输出相同
    torch.backends.cudnn.benchmark = False  # 禁用自动优化功能


# 细化每个细胞的标签: 考虑其周围 radius 个最近邻的标签，通过选择邻居中最常见的标签来更新细胞的标签
def refine_label(adata, radius=25, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]

    return new_type


def mclust_R(andata, seed, num_cluster=7, modelNames='EEE', used_obsm='z'):
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r.library("mclust")

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(andata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    andata.obs['mclust'] = mclust_res
    andata.obs['mclust'] = andata.obs['mclust'].astype('int')
    andata.obs['mclust'] = andata.obs['mclust'].astype('category')
    return andata


def plot_loss(pretrain_epoch, loss_list, CA_epochs, embs_CA1, embs_CA2, png_path):
    # 绘制曲线
    epochs = range(len(loss_list))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch (s)')
    ax1.plot(epochs, loss_list, 'r', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(CA_epochs, embs_CA1, 'g', label='mclust ARI')
    ax2.plot(CA_epochs, embs_CA2, 'b', label='mclust NMI')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.axvline(x=pretrain_epoch, ls="--", color="black")  # 竖线
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.savefig(png_path + 'acc.png')


def kmcluster(adata, z, seed, num_cluster, labels, png_path, epoch, dataset_name, refine=False,
              dataset_class="original_clusters"):
    emb = z.detach().cpu().numpy()
    adata.obsm['emb'] = emb
    # z = np.array((z / z.norm(dim=1)[:, None]).detach().cpu().numpy())

    print("cluster num is set to {}".format(num_cluster))
    # kmeans  归一化的数据聚类
    # cluster_method1 = KMeans(n_clusters=num_cluster, random_state=seed)
    # pd_labels1 = cluster_method1.fit(z).labels_
    # embs_eval_kmeans = compute_metrics(emb, labels, pd_labels1)

    # mclust  使用未归一化的数据聚类
    cluster_data = sc.AnnData(emb)
    cluster_data.obsm['z'] = emb
    mclust_R(cluster_data, seed, num_cluster, used_obsm='z')
    pd_labels2 = cluster_data.obs['mclust'].astype(int).to_numpy()
    embs_eval_mclust = compute_metrics(emb, labels, pd_labels2)

    adata.obs['mclust'] = pd_labels2
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    new2 = refine_label(adata, key='mclust')
    adata.obs['mclust_r'] = new2
    embs_eval_refine_m = compute_metrics(emb, labels, np.array(new2).astype(int))

    sc.pp.neighbors(adata, use_rep='emb', n_neighbors=30)
    sc.tl.umap(adata)
    sc.tl.paga(adata, groups=dataset_class)

    if epoch == 'pretrain-final-' or epoch == 'train-final-':
        # 创建画布，1 行 2 列的子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        if 'spatial' not in adata.uns.keys():
            sc.pl.embedding(adata, basis="spatial", color=dataset_class, title=['Ground Truth'], ax=ax1, show=False)
            sc.pl.embedding(adata, basis="spatial", color="mclust",
                            title=['mclust-ARI=%.3f' % (embs_eval_mclust["ARI"])], ax=ax2, show=False)
            sc.pl.embedding(adata, basis="spatial", color=["mclust_r"],
                            title=['mclust refine-ARI=%.3f' % (embs_eval_refine_m["ARI"])], ax=ax3, show=False)
        else:
            sc.pl.spatial(adata, color=[dataset_class], title=['Ground Truth'], ax=ax1, show=False)
            sc.pl.spatial(adata, color=["mclust"], title=['mclust-ARI=%.3f' % (embs_eval_mclust["ARI"])], ax=ax2,
                          show=False)
            sc.pl.spatial(adata, color=["mclust_r"], title=['mclust refine-ARI=%.3f' % (embs_eval_refine_m["ARI"])],
                          ax=ax3, show=False)
        # 调整布局
        plt.tight_layout()
        plt.savefig(png_path + "-" + str(epoch) + "spatial.png")
        # 关闭图像以释放内存
        plt.close(fig)

        # 创建画布，1 行 2 列的子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))  # dlpfc 18/6
        sc.pl.umap(adata, color=dataset_class, title=[dataset_name + 'embedding_umap-truth'], ax=ax1, show=False,
                   size=30)

        sc.pl.paga(adata, color=dataset_class, title=dataset_name + '_paga', ax=ax2, show=False, fontsize=12)
        # 调整布局
        plt.tight_layout()
        plt.savefig(png_path + "-" + str(epoch) + "analyze.png")
        # 关闭图像以释放内存
        plt.close(fig)

    if refine:
        adata.obs['domain'] = adata.obs['mclust_r']
        evals = embs_eval_refine_m
    else:
        adata.obs['domain'] = adata.obs['mclust']
        evals = embs_eval_mclust

    return embs_eval_mclust, embs_eval_refine_m, evals


# -------------------adj-------------------
#  DLPFC的r=150，HBCA1的r=300，主要目的是使邻居均值为5-6
# 计算邻接矩阵
def Cal_Spatial_Net(adata, rad_cutoff=150, max_neigh=50, verbose=True):
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    n_spot = coor.shape[0]

    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(
        coor)  # 先选出50个最近邻居并且将保存源节点-目标节点-距离的矩阵
    distances, indices = nbrs.kneighbors(
        coor)  # 返回两个节点数*（max_neigh+1）的矩阵，distances返回i节点和最近的50个节点的距离，indices返回i节点和最近的50个节点的索引
    interaction1 = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        interaction1[i, indices[i]] = 1
    adata.obsm['graph_neigh'] = interaction1  # 自环的

    indices = indices[:, 1:]  # 删除i自身
    distances = distances[:, 1:]  # 删除i自身

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']  # 给每列index
    Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]  # 筛选150以内的点
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), )).copy()
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], coor.shape[0]))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / coor.shape[0]))

    adata.uns['Spatial_Net'] = Spatial_Net  # cell1,cell2,distance


# 构建邻接稀疏矩阵
def Transfer_Spatial_Net(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)  # 通过obs获取barcodes序列，转化为np.array，shape->(4226，)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))  # 打包为一个dic，keys为细胞名，values为其索引
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)  # 进行map映射
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # 构建稀疏矩阵,sp.coo_matrix(data,(row,col),shape(N*N)),有连接的置为1，shape->(16904,16904)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # 1&0  添加自环,对角线全为1
    adata.obsm['adj'] = G  # 稀疏


# 加对比标注
def add_contrastive_label(adata):
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


# 将给定的邻接矩阵进行归一化   D^(-1/2)*A*D^(-1/2)
def normalize_adj(adj_):
    adj_ = sp.coo_matrix(adj_)
    adj = adj_ + sp.eye(adj_.shape[0])  # 添加自环,对角线加1
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj


# =======ZINB loss start==========#
def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


# =======ZINB loss end==========#

def get_matrix(pi, disp, mean):
    reconstructed = (1 - pi) * mean  # 形状: (n_cells, n_genes)
    return reconstructed

# 一致性损失 衡量两个嵌入向量组（emb1 和 emb2）之间的相似性
def consistency_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)  # 中心化（减去均值）
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)  # L2归一化（单位向量）
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())  # emb1的协方差矩阵（Gram矩阵）
    cov2 = torch.matmul(emb2, emb2.t())
    return torch.mean((cov1 - cov2) ** 2)  # 协方差矩阵的均方误差（MSE）


# 拓扑结构损失
def adjacency_loss(rec_adj, adj):
    pos_weight = torch.tensor([adj.shape[0] ** 2 / (adj.shape[0] ** 2 - adj.sum())]).to(adj.device)
    return F.binary_cross_entropy_with_logits(rec_adj, adj, pos_weight=pos_weight)


# =============================== 计算各种评价指标 =================================#
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed  计算聚类ACC
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values

    # CJY 2021.6.20 for batch effect dataset
    # y_true = y_true.to_list()
    y_pred = y_pred.astype(np.int64)

    label_to_number = {label: number for number, label in enumerate(set(y_true))}
    label_numerical = np.array([label_to_number[i] for i in y_true])

    y_true = label_numerical.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    # ind = linear_assignment(w.max() - w)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    # https://stackoverflow.com/questions/62390517/no-module-named-sklearn-utils-linear-assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    row_ind, col_ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


# 2. Jaccard score (JS)
# Please refer to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
def Jaccard_index(y_true, y_pred):
    from sklearn.metrics.cluster import pair_confusion_matrix
    contingency = pair_confusion_matrix(y_true, y_pred)
    JI = contingency[1, 1] / (contingency[1, 1] + contingency[0, 1] + contingency[1, 0])
    return JI


def compute_metrics(embs, y_true, y_pred):
    metrics = {}
    metrics["ARI"] = round(ARI(y_true, y_pred), 3)
    metrics["NMI"] = round(NMI(y_true, y_pred), 3)
    metrics["CA"] = round(cluster_acc(y_true, y_pred), 3)
    metrics["JI"] = round(Jaccard_index(y_true, y_pred), 3)
    metrics["CS"] = round(completeness_score(y_true, y_pred), 3)
    metrics["HS"] = round(homogeneity_score(y_true, y_pred), 3)
    metrics["VMS"] = round(v_measure_score(y_true, y_pred), 3)
    metrics["Sil"] = round(silhouette_score(embs, y_pred), 3)
    metrics["CH"] = round(calinski_harabasz_score(embs, y_pred), 3)

    return metrics
