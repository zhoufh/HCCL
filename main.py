# Created by Zhoufanghui at 2025/7/27
import argparse
import logging
import os
import warnings

import numpy as np
import torch

warnings.simplefilter(action='ignore')
os.environ["OMP_NUM_THREADS"] = '1'
from model.train import runDemo
import st_loadUtils as loadUtils


def get_args(dataset_name, dataset_class, n_cluster, png_path, log_path, pretrain_path, a, b, c, d) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--dataset_name", type=str, default=dataset_name)
    parser.add_argument("--dataset_class", type=str, default=dataset_class,
                        help='labels for data holding classified information')
    parser.add_argument("--n_cluster", type=int, default=n_cluster)
    parser.add_argument('--radius', type=int, default=150, help='to generate the adjacency matrix')
    parser.add_argument("--refine", type=bool, default=True)
    parser.add_argument("--reconstructed", type=bool, default=False)
    parser.add_argument('--png_path', type=str, default=png_path)
    parser.add_argument('--log_path', type=str, default=log_path)
    parser.add_argument('--train_path', type=str, default=pretrain_path,
                        help='Save the model after pretraining has finished. ')

    parser.add_argument("--in_channels", default=3000, type=int)
    parser.add_argument('--train_epochs', default=401, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--eval_freq', default=50, type=int, metavar='N',
                        help='Save frequency (default: 20)')
    parser.add_argument('--train_lr', default=0.001, type=float,
                        help='initial learning rate of pretrain')
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument('--lambda1', default=a, type=float, help='loss_rec weight')
    parser.add_argument('--lambda2', default=b, type=float, help='loss_sl_1 weight')
    parser.add_argument('--lambda3', default=c, type=float, help='loss_cl weight')
    parser.add_argument('--lambda4', default=d, type=float, help='loss_adj1 weight')

    args = parser.parse_args()
    return args


def get_args_key(args):
    return "-".join([args.dataset_name])


def pprint_args(_args: argparse.Namespace):
    print("Args PPRINT: {}".format(get_args_key(_args)))
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))


if __name__ == '__main__':
    dataset_name = ['151507', '151508', '151509', '151510', '151669', '151670',
                    '151671', '151672', '151673', '151674', '151675', '151676']
    root_path = "result/"
    log_name = 'github'
    os.environ['R_HOME'] = 'D:/SoftWare/R_Language/R-4.3.2'
    dataset_class = 'original_clusters'

    loss_params = {'a': 1, 'b': 1.5, 'c': 0.5, 'd': 0.1}

    ari_list = []
    refine_list = []
    final_list = []

    param_str = f"params_{loss_params['a']}_{loss_params['b']}_{loss_params['c']}_{loss_params['d']}"
    for name in dataset_name[0:1]:
        print("## Loading Dataset {}##".format(name))
        adata = loadUtils.load_DLPFC(section_id=name)  # 加载存储原始基因表达矩阵的对象
        num_cluster = len(list(set(adata.obs[dataset_class])))

        log_path = root_path + name + log_name + "-" + param_str + ".txt"
        png_path = root_path + name + log_name + "-" + param_str + "-"
        pretrain_path = root_path + name + log_name + "-" + param_str + "_model.pth"
        main_args = get_args(
            # 数据集相关信息  存储文件路径
            dataset_name=name, dataset_class=dataset_class, n_cluster=num_cluster,
            png_path=png_path, log_path=log_path, pretrain_path=pretrain_path,
            # 传入当前随机采样的参数
            **loss_params
        )
        pprint_args(main_args)
        logging.basicConfig(level=logging.INFO,
                            filename=log_path,
                            filemode='w',
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        adata, ari, refine, final = runDemo(args=main_args, adata=adata)
        ari_list.append(ari)
        final_list.append(final)
        refine_list.append(refine)
        save_path = os.path.join(root_path, name+"_"+param_str+"_final.h5ad")
        adata.write(save_path)
        print("Done!!!!")

    logging.info(f"param_str: {param_str} --------")
    logging.info(f"mean-ari: {np.mean(ari_list)} | mari: {ari_list}")
    logging.info(f"refinemean-ari: {np.mean(refine_list)} | refine mari: {refine_list}")
    logging.info(f"finalmean-ari: {np.mean(final_list)} | final ari: {final_list}")
    logging.info("-" * 30)

    print(f"param_str: {param_str} --------")
    print(f"mean-ari: {np.mean(ari_list)} | mari: {ari_list}")
    print(f"refinemean-ari: {np.mean(refine_list)} | refine mari: {refine_list}")
    print(f"finalmean-ari: {np.mean(final_list)} | final ari: {final_list}")
