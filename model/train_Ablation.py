# Created by Zhoufanghui at 2025/7/29
# Created by Zhoufanghui at 2025/7/22
import logging
import time

import torch.utils.data
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .GCL_loader import STDataset
from .GCL_model import ConCH
from .utils import *


def runDemo(args, adata):
    fix_seed(args.seed)
    # 数据增强参数
    args_transformation = [
        {
            'noise_percentage': 0.8, 'sigma': 0.5, 'apply_noise_prob': 1  # (Add) gaussian noise
        }, {
            'shuffle_non_hvg': 0.4,   # 非HVG扰动概率
            'dropout_non_hvg': 0.5,   # 非HVG扰动概率
            'hvg_perturb_prob': 0.4,  # HVG扰动概率
            'strategy_weights': [0.65, 0.35]    # 策略分布概率
        }]

    logging.info('args: {}'.format(args_transformation))
    adata = preprocess(adata)   # 数据预处理
    if 'adj' not in adata.obsm.keys():  # 计算图结构
        Cal_Spatial_Net(adata, rad_cutoff=args.radius)
        Transfer_Spatial_Net(adata)
    add_contrastive_label(adata)  # 加一个对比标注矩阵
    print(adata)
    num_cluster = args.n_cluster
    num_cell, num_feature = adata.X.shape
    print(f"Train dataset num cells: {num_cell}, num features: {num_feature}")  # feature is inputdim
    indices = np.arange(num_cell)

    #--------------数据准备------------------------
    train_data = STDataset(adata=adata, obs_label_colname=args.dataset_class,
                            args_transformation_list=args_transformation, normal_shuffle=False, index_list=indices)
    evaldata, labels = train_data.getEval()
    evaldata = torch.FloatTensor(evaldata).to(args.device)    # sample_h
    adj = torch.FloatTensor(adata.obsm['adj'].toarray()).to(args.device)
    norm_adj = torch.FloatTensor(normalize_adj(adata.obsm['adj']).toarray()).to(args.device)  # 邻接矩阵标准化
    graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(adj.shape[0])).to(args.device)
    label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(args.device)

    # -------------model preparation-------------
    train_model = ConCH(num_feature, n_clusters=num_cluster, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.Adam(train_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)  # 优化器
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    # --------------model train------------------
    loss_list = []
    mclust_NMI = []
    mclust_ARI = []
    ARI_epochs = []
    earlystop_count = 0  # 用于统计是否要早停
    last_loss = 0  # 储存训练的上一次的损失值，用于统计当前损失比上一次降低了多少
    epoch_num = 0
    min_loss = 1e4  # 用于储存当前训练过程中最小的损失值
    start_time = time.time()
    print('train model：')
    logging.info("+" * 10 + args.dataset_name + "+" * 10)
    logging.info("pretrain_loss = {}*loss_rec + {}*loss_sl_1 + {}*loss_cl + {}*loss_adj1"
                 .format(args.lambda1, args.lambda2, args.lambda3, args.lambda4))
    BCELoss = nn.BCEWithLogitsLoss()
    for current_iter, epoch in enumerate(tqdm(range(args.train_epochs))):
        # Load data
        samples = train_data.getFeature()
        feat, feat_a, feat_b = [v.to(args.device) for v in samples]

        torch.set_grad_enabled(True)
        train_model.train()
        optimizer.zero_grad()
        z1, z2, z3, pi, disp, mean, rec_adj, ret1 = train_model(feat, feat_a, feat_b, norm_adj, graph_neigh)
        # -----重构损失-------
        loss_rec = ZINB(pi, theta=disp, ridge_lambda=0).loss(feat, mean, mean=True)
        # -----拓扑结构重构损失-----
        loss_adj1 = adjacency_loss(rec_adj, adj)
        # -----对比损失-----
        loss_cl = BCELoss(ret1, label_CSL)  # 局部-全局互信息最大化  DGI的损失函数
        # -----一致性损失--------
        loss_con = consistency_loss(z1, z3)
        # -----总损失----------
        loss = args.lambda1 * loss_rec + args.lambda2 * loss_cl + args.lambda3 * loss_con + args.lambda4 * loss_adj1
        loss_list.append(loss.cpu().detach().numpy())
        gap = loss - last_loss
        last_loss = loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        # if epoch % args.eval_freq == 0 and epoch != 0:
        #     print('Pretrain Epoch {}, Loss_all {}, Loss recon {} loss_sl_1 {} loss_cl {} loss_adj1 {}'
        #           .format(epoch, loss, loss_rec, loss_cl, loss_con, loss_adj1))
        #     print("Pretrain_epoch {}\tgap {}\tpretrein_lr {}".format(epoch, gap, scheduler.get_last_lr()[0]))
        #
        #     train_model.eval()
        #     epoch_num += 1
        #     z = train_model(evaldata, evaldata, evaldata, norm_adj.clone(), graph_neigh.clone())[0]
        #     emb = z.detach().cpu().numpy()  # 原始emb
        #     # z = np.array((z / z.norm(dim=1)[:, None]).detach().cpu().numpy())  # 归一化emb
        #     print("cluster num is set to {}".format(num_cluster))
        #     # kmeans
        #     # pd_labels1 = KMeans(n_clusters=num_cluster, random_state=args.seed).fit(z).labels_
        #     # embs_eval_kmeans = compute_metrics(emb, labels, pd_labels1)
        #     # mclust
        #     cluster_data = sc.AnnData(emb)
        #     cluster_data.obsm['z'] = emb
        #     mclust_R(cluster_data, args.seed, num_cluster, used_obsm='z')
        #     pd_labels2 = cluster_data.obs['mclust'].astype(int).to_numpy()
        #     embs_eval_mclust = compute_metrics(emb, labels, pd_labels2)
        #     mclust_NMI.append(embs_eval_mclust["NMI"])
        #     # KMeans_ARI.append(embs_eval_kmeans["ARI"])
        #     mclust_ARI.append(embs_eval_mclust["ARI"])
        #     ARI_epochs.append(epoch)
        #     print('++++++++++++++++++++++++++')
        #     # print("Pretrain Epoch: {}\t kmeans {}".format(epoch, embs_eval_kmeans))
        #     print("Pretrain Epoch: {}\t mclust {}".format(epoch, embs_eval_mclust))
        #     # logging.info("Pretrain Epoch: {}\t kmeans {}".format(epoch, embs_eval_kmeans))
        #     logging.info("Pretrain Epoch: {}\t mclust {}".format(epoch, embs_eval_mclust))
        # # train_model.train()

    torch.save(train_model.state_dict(), args.train_path)
    end_time = time.time()
    print('Elapsed training time:{:.4f} seconds'.format((end_time - start_time)))
    print("model saved to {}.".format(args.train_path))
    plot_loss(args.train_epochs, loss_list, ARI_epochs, mclust_ARI, mclust_NMI, args.png_path)

    # --------------model eval------------------
    new_model = ConCH(num_feature, n_clusters=num_cluster, dropout=args.dropout).to(args.device)
    new_model.load_state_dict(torch.load(args.train_path))
    new_model.eval()

    z, _, _, pi, disp, mean, _, _ = new_model(evaldata, evaldata, evaldata, norm_adj, graph_neigh)
    if args.reconstructed:
        print("yes")
        adata.obsm['X_reconstructed'] = get_matrix(pi,disp,mean).detach().cpu().numpy()
    embs_eval_mclust, embs_eval_refine_m, evals = kmcluster(adata, z, args.seed, num_cluster, labels, args.png_path,
                                                            refine=args.refine, epoch="train-final-",
                                                            dataset_name=args.dataset_name)

    # 训练完之后的模型
    print(f"dataset {args.dataset_name} train final\tmclust {embs_eval_mclust}")
    print(f"dataset {args.dataset_name} refine final\tmclust {embs_eval_refine_m}")
    print(f"dataset {args.dataset_name} final domain: {evals}")
    logging.info(f"-------dataset {args.dataset_name} train final\tmclust {embs_eval_mclust}")
    logging.info(f"-------dataset {args.dataset_name} refine final\tmclust {embs_eval_refine_m}")

    return adata, embs_eval_mclust["ARI"], embs_eval_refine_m["ARI"], evals["ARI"]
