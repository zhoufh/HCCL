# Created by Zhoufanghui at 2025/7/22
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


class STDataset:
    def __init__(self,
                 adata,
                 obs_label_colname,  # ground truth列名
                 args_transformation_list,  # 变换参数
                 index_list,  # 索引
                 normal_shuffle=False  # 是否是消融实验
                 ):
        self.adata = adata
        if isinstance(self.adata.X, np.ndarray):  # 数据类型转换
            self.data = self.adata.X
        else:
            self.data = self.adata.X.toarray()
        self.dataset_for_transform = deepcopy(self.data)  # 用于变换的数据

        if self.adata.obs.get(obs_label_colname) is not None:  # 标签处理
            self.label = self.adata.obs[obs_label_colname]
            self.unique_label = list(set(self.label))

            self.label_encoder = {k: v for k, v in zip(self.unique_label, range(len(self.unique_label)))}
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}
        else:
            self.label = None
            print("Can not find corresponding labels")

        # do the transformation
        self.normal_shuffle = normal_shuffle  # 消融实验用的，True是使用随机打乱
        self.num_cells, self.num_genes = self.adata.shape
        self.args_transformation_list = args_transformation_list  # 扰动参数

        self.index_list = index_list
        self.sample_h = self.dataset_for_transform[index_list].astype(np.float32)
        self.tr = transformation(self.adata, self.sample_h)
        if self.normal_shuffle is False:  # 不是消融实验
            # 初始运行，找到参数
            current_args = self.tr.generate_high_similarity_with_threshold(self.args_transformation_list[1])
            self.args_transformation_list[1] = current_args  # 更新参数

    def getFeature(self):
        sample_a = self.tr.gaussion_transform(self.args_transformation_list[0])  # gaussion 变换样本
        if self.normal_shuffle:  # 消融实验
            sample_b = self.tr.shuffledata()
        else:  # high_similarity 变换样本,使用训练好的参数
            sample_b = self.tr.high_similarity_transform(self.args_transformation_list[1])

        sample_h = torch.FloatTensor(self.sample_h)
        sample_a = torch.FloatTensor(sample_a)
        sample_b = torch.FloatTensor(sample_b)

        return [sample_h, sample_a, sample_b]

    def getEval(self):

        labels = np.array([
            self.label_encoder.get(self.label[index], -1)
            if self.label is not None else -1
            for index in self.index_list
        ], dtype=np.int32)

        return self.sample_h, labels


class transformation:
    def __init__(self,
                 dataset,  # adata
                 sample,   # 变换矩阵
                 target_sim=0.8
                 ):
        self.dataset = dataset
        self.sample = deepcopy(sample)
        self.n_samples, self.n_features = self.sample.shape
        self.target_sim = target_sim

        highly_variable = self.dataset.var['top_highly_variable'].values  # 找关键基因
        self.hvg_indices = np.where(highly_variable)[0]
        self.non_hvg_indices = np.where(~highly_variable)[0]

    def shuffledata(self):
        # 随机打乱，按照cell打乱
        transformed_samples = self.sample.copy()
        ids = np.arange(self.n_samples)
        ids = np.random.permutation(ids)
        feature_permutated = transformed_samples[ids]
        return feature_permutated

    def gaussion_transform(self, args_transformation):
        # 添加高斯噪声
        transformed_samples = self.sample.copy()
        s = np.random.uniform(0, 1)

        if args_transformation['apply_noise_prob'] > s:
            noise_percentage = args_transformation['noise_percentage']
            sigma = args_transformation['sigma']

            noise_prob = np.random.rand(self.n_samples, self.n_features) < noise_percentage

            noise = np.random.normal(0, sigma, size=(self.n_samples, self.n_features))

            transformed_samples[noise_prob] += noise[noise_prob]

        return transformed_samples

    def high_similarity_transform(self, args_transformation):
        # 生成高相似度负样本
        transformed_samples = self.sample.copy()

        for i in range(self.n_samples):
            # 选择变换策略（带动态权重）
            # 部分打乱 局部遮蔽
            strategy = np.random.choice(['partial_shuffle', 'local_dropout'],
                                        p=args_transformation['strategy_weights'])

            if strategy == 'partial_shuffle':  # 部分打乱
                # 非HVG打乱
                n_shuffle = int(len(self.non_hvg_indices) *  args_transformation['shuffle_non_hvg'])  # 打乱非高变基因
                if n_shuffle > 0:
                    shuffle_genes = np.random.choice(self.non_hvg_indices, n_shuffle, replace=False)
                    transformed_samples[i, shuffle_genes] = np.random.permutation(transformed_samples[i, shuffle_genes])

                # 以概率对HVG进行轻度打乱
                if np.random.rand() < args_transformation['hvg_perturb_prob']:
                    n_hvg_shuffle = int(len(self.hvg_indices) * 0.1)  # 少量HVG打乱
                    if n_hvg_shuffle > 0:
                        shuffle_hvg = np.random.choice(self.hvg_indices, n_hvg_shuffle, replace=False)
                        transformed_samples[i, shuffle_hvg] = np.random.permutation(transformed_samples[i, shuffle_hvg])

            elif strategy == 'local_dropout':  # 局部遮蔽
                # 非HVG dropout
                non_hvg_dropout = (np.random.rand(len(self.non_hvg_indices)) < args_transformation['dropout_non_hvg'])
                transformed_samples[i, self.non_hvg_indices[non_hvg_dropout]] = 0

                # HVG部分dropout（概率较低）
                if np.random.rand() < args_transformation['hvg_perturb_prob'] / 2:
                    hvg_dropout = (np.random.rand(len(self.hvg_indices)) < 0.1)
                    transformed_samples[i, self.hvg_indices[hvg_dropout]] = 0

            elif strategy == 'targeted_noise':
                # 差异化噪声   高变基因弱噪声(σ×0.1)，非高变基因强噪声(σ×0.5)
                hvg_noise = np.random.normal(0, args_transformation['sigma_hvg'], size=len(self.hvg_indices))
                non_hvg_noise = np.random.normal(0, args_transformation['sigma_non_hvg'],
                                                 size=len(self.non_hvg_indices))
                transformed_samples[i, self.hvg_indices] += hvg_noise
                transformed_samples[i, self.non_hvg_indices] += non_hvg_noise

        return transformed_samples

    def generate_high_similarity_with_threshold(self, args_transformation, max_attempts=10):
        # 包装函数，通过迭代调用high_similarity_transform生成相似度小于阈值的样本
        # max_attempts: 最大尝试次数
        # 创建参数的深拷贝以避免修改原始参数
        current_args = {
            'shuffle_non_hvg': args_transformation.get('shuffle_non_hvg', 0.4),
            'dropout_non_hvg': args_transformation.get('dropout_non_hvg', 0.5),
            'hvg_perturb_prob': args_transformation.get('hvg_perturb_prob', 0.4),
            'strategy_weights': args_transformation.get('strategy_weights', [0.6, 0.4]),
        }

        for attempt in range(max_attempts):
            # 调用原始变换函数
            transformed = self.high_similarity_transform(current_args)

            # 计算相似度
            sims = np.array([cosine_similarity(self.sample[i:i + 1], transformed[i:i + 1])[0, 0]
                             for i in range(self.n_samples)])
            avg_sim = np.mean(sims)

            print(f"Attempt {attempt + 1}: 平均相似度 = {avg_sim:.3f}")
            # 检查是否满足条件
            if avg_sim < self.target_sim:
                print(f"成功生成相似度 {avg_sim:.3f} < {self.target_sim} 的样本")
                print(f"当前参数: {current_args}")
                return current_args

            # 如果不满足，增强扰动参数
            current_args['shuffle_non_hvg'] = min(0.8, current_args['shuffle_non_hvg']*1.2)
            current_args['dropout_non_hvg'] = min(0.8, current_args['dropout_non_hvg']*1.2)
            current_args['hvg_perturb_prob'] = min(0.8, current_args['hvg_perturb_prob'] * 1.2)  # 增加HVG扰动概率
            # 调整策略权重，增加更激进策略的权重
            current_args['strategy_weights'] = [
                max(0.1, current_args['strategy_weights'][0] * 0.9),  # 减少partial_shuffle
                min(0.5, current_args['strategy_weights'][1] * 1.1)  # 增加local_dropout
            ]
            # 归一化权重
            total = sum(current_args['strategy_weights'])
            current_args['strategy_weights'] = [w / total for w in current_args['strategy_weights']]

        print(f"警告: 在{max_attempts}次尝试后未能达到目标相似度，返回最后一次结果")
        print(f"最终参数: {current_args}")
        return current_args
