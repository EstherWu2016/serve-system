import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

class LightGCNConv(MessagePassing):
    """
    LightGCN卷积层实现
    简化了GraphSAGE和GCN，仅保留邻居聚合，去掉转换矩阵和非线性激活函数
    """
    def __init__(self, **kwargs):
        super(LightGCNConv, self).__init__(aggr='add', **kwargs)

    def forward(self, x, edge_index):
        # 计算节点的度数（每个节点连接的边数量）用于归一化
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # 边权重归一化
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 开始消息传递/聚合
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # 消息传递函数
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 不做任何更新操作，直接返回聚合的结果
        return aggr_out


class LightGCN(nn.Module):
    """
    LightGCN模型实现
    """
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3, dropout=0.1):
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 初始化用户和物品嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 使用正态分布初始化嵌入权重
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # LightGCN卷积层
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        
    def forward(self, edge_index):
        """
        前向传播函数
        
        参数:
            edge_index: 边索引张量，形状为 [2, num_edges]
        
        返回:
            user_emb: 最终的用户嵌入
            item_emb: 最终的物品嵌入
        """
        # 获取初始嵌入
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # 将用户和物品嵌入合并
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # 保存每一层的嵌入，用于最后的跳跃连接
        all_embs = [x]
        
        # 图卷积传播
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            all_embs.append(x)
        
        # 层间跳跃连接 (mean pooling)
        all_embs = torch.stack(all_embs, dim=0)
        all_embs = all_embs.mean(dim=0)
        
        # 分离用户和物品嵌入
        user_emb, item_emb = torch.split(all_embs, [self.num_users, self.num_items])
        
        return user_emb, item_emb
    
    def bpr_loss(self, users, pos_items, neg_items):
        """
        计算贝叶斯个性化排序(BPR)损失
        
        参数:
            users: 用户索引
            pos_items: 正样本物品索引
            neg_items: 负样本物品索引
            
        返回:
            loss: BPR损失
        """
        # 获取用户和物品的嵌入
        device = users.device
        edge_index = self.edge_index.to(device)
        user_emb, item_emb = self.forward(edge_index)
        
        # 获取指定用户和物品的嵌入
        user_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # 计算正样本和负样本的评分
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        
        # BPR损失
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-5).mean()
        
        # L2正则化
        regularization = 0.5 * (user_emb.norm(2).pow(2) + 
                                pos_emb.norm(2).pow(2) + 
                                neg_emb.norm(2).pow(2)) / len(users)
        
        return loss, regularization
    
    def predict(self, users):
        """
        为指定用户预测所有物品的评分
        
        参数:
            users: 用户索引列表
            
        返回:
            scores: 用户对所有物品的预测评分，形状为 [len(users), num_items]
        """
        # 获取所有用户和物品的嵌入
        user_emb, item_emb = self.forward(self.edge_index)
        
        # 获取指定用户的嵌入
        user_emb = user_emb[users]
        
        # 预测评分 (用户嵌入与所有物品嵌入的内积)
        scores = torch.matmul(user_emb, item_emb.t())
        
        return scores
    
    def set_edge_index(self, edge_index):
        """
        设置图的边索引
        
        参数:
            edge_index: 边索引张量，形状为 [2, num_edges]
        """
        self.edge_index = edge_index 