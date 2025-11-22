# Created by Zhoufanghui at 2025/7/22
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, self.training)
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class Discriminator(nn.Module):
    # 判别器
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  # 扩展维度

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


# 两层   nfeat 3000  nhid1 = 128  nhid2 = 64
class ZINBDecoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(ZINBDecoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class ConCH(nn.Module):
    def __init__(self, in_channels, n_clusters=15,
                 hidden_dims=[500, 64],
                 dropout=0.01,
                 alpha=1.5,
                 act=F.relu
                 ):
        super(ConCH, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.n_clusters = n_clusters
        self.dropout = dropout
        self.alpha = alpha
        self.act = act
        self.sig = nn.Sigmoid()
        self.disc = Discriminator(hidden_dims[-1])

        self.conv1 = GraphConvolution(in_channels, hidden_dims[0], self.dropout, act=F.relu)  # 3000-500
        self.conv2 = GraphConvolution(hidden_dims[0], hidden_dims[-1], self.dropout, act=lambda x: x)  # 500-64

        self.ZINB = ZINBDecoder(in_channels, hidden_dims[0], hidden_dims[-1])

    def encode(self, x, adj):
        x = self.conv1(x, adj)
        h = self.conv2(x, adj)
        return h

    def read_out(self, emb, mask=None):
        '''
            加权平均聚合：使用掩码矩阵 mask 对输入嵌入 emb 进行加权平均。
            mask 的每一行表示一个样本中各个节点的权重。
            计算每个样本的全局嵌入表示，并对结果进行 L2 范数归一化。
        '''
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)

    def dot_product_decode(self, z):
        # 实现了一个简单的内积解码器，用于通过节点特征 z 计算节点之间的相似性或连接预测
        # 重构了邻接矩阵
        z = F.normalize(z, p=2, dim=1)  # 按行归一化
        rec_adj = torch.sigmoid(torch.matmul(z, z.t()))
        return rec_adj

    def forward(self, feat, feat_a, feat_b, adj, graph_neigh):
        z1 = self.encode(feat, adj)
        emb1 = self.act(z1)
        # 重构
        [pi, disp, mean] = self.ZINB(z1)
        rec_adj = self.dot_product_decode(z1)
        # pos data
        z2 = self.encode(feat_a, adj)
        emb2 = self.act(z2)
        # 重构
        # [pi, disp, mean] = self.ZINB(z2)
        # rec_adj = self.dot_product_decode(z2)

        # neg data
        z3 = self.encode(feat_b, adj)
        emb3 = self.act(z3)

        g2 = self.read_out(emb1, graph_neigh)
        g2 = self.sig(g2)       # 正样本的全局表示

        g3 = self.read_out(emb3, graph_neigh)
        g3 = self.sig(g3)  # 正样本的全局表示

        ret1 = self.disc(g2, emb1, emb3)
        # ret2 = self.disc(g3, emb3, emb1)

        return z1, z2, z3, pi, disp, mean, rec_adj, ret1
