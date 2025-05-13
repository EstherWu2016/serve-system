import torch
import torch.nn as nn
import torch.nn.functional as F

class RankNet(nn.Module):
    """
    RankNet模型实现
    基于pairwise的排序学习，使用物品对来训练模型
    """
    def __init__(self, user_feat_dim, item_feat_dim, embedding_dim=64, hidden_dims=[128, 64, 32], dropout=0.2):
        super(RankNet, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 用户嵌入层
        self.user_embedding = nn.Linear(user_feat_dim, embedding_dim)
        
        # 物品嵌入层
        self.item_embedding = nn.Linear(item_feat_dim, embedding_dim)
        
        # MLP层
        mlp_layers = []
        input_dim = embedding_dim * 2  # 用户向量 + 物品向量
        
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, user_features, item_features):
        """
        前向传播函数
        
        参数:
            user_features: 用户特征 [batch_size, user_feat_dim]
            item_features: 物品特征 [batch_size, item_feat_dim]
        
        返回:
            scores: 物品的排序分数 [batch_size, 1]
        """
        # 转换为嵌入向量
        user_emb = self.user_embedding(user_features)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_features)  # [batch_size, embedding_dim]
        
        # 拼接特征
        concat_features = torch.cat([user_emb, item_emb], dim=1)
        
        # 通过MLP层
        mlp_output = self.mlp(concat_features)
        
        # 输出层
        scores = self.output_layer(mlp_output)
        
        return scores.squeeze(-1)
    
    def calculate_pairwise_loss(self, user_features, pos_item_features, neg_item_features, sigma=1.0):
        """
        计算RankNet损失
        
        参数:
            user_features: 用户特征 [batch_size, user_feat_dim]
            pos_item_features: 正样本物品特征 [batch_size, item_feat_dim]
            neg_item_features: 负样本物品特征 [batch_size, item_feat_dim]
            sigma: 控制激活函数的平滑程度
            
        返回:
            loss: RankNet损失
        """
        # 计算正样本和负样本的分数
        pos_scores = self.forward(user_features, pos_item_features)  # [batch_size]
        neg_scores = self.forward(user_features, neg_item_features)  # [batch_size]
        
        # 计算分数差异
        diff = sigma * (pos_scores - neg_scores)
        
        # 使用交叉熵损失，其中目标是正样本比负样本的分数高
        loss = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
        
        return loss
    
    def get_rank_scores(self, user_features, item_features_list):
        """
        获取多个物品的排序分数
        
        参数:
            user_features: 用户特征 [batch_size, user_feat_dim]
            item_features_list: 候选物品特征列表，每个元素形状为 [batch_size, item_feat_dim]
            
        返回:
            scores: 每个物品的排序分数 [batch_size, num_items]
        """
        batch_size = user_features.size(0)
        num_items = len(item_features_list)
        
        # 初始化分数矩阵
        scores = torch.zeros(batch_size, num_items, device=user_features.device)
        
        # 为每个物品计算分数
        for i, item_features in enumerate(item_features_list):
            scores[:, i] = self.forward(user_features, item_features)
        
        return scores 