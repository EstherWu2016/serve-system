import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    注意力层实现
    用于计算用户历史物品与候选物品的相关性权重
    """
    def __init__(self, embedding_dim, attention_size):
        super(Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_size = attention_size
        
        # 注意力网络参数
        self.w_query = nn.Linear(embedding_dim, attention_size, bias=False)
        self.w_key = nn.Linear(embedding_dim, attention_size, bias=False)
        self.w_value = nn.Linear(attention_size, 1, bias=False)
        
    def forward(self, queries, keys, keys_length):
        """
        计算注意力权重并返回加权后的表示
        
        参数:
            queries: 候选物品嵌入向量 [batch_size, embedding_dim]
            keys: 历史物品嵌入向量序列 [batch_size, max_seq_len, embedding_dim]
            keys_length: 每个用户的历史序列实际长度 [batch_size]
        
        返回:
            output: 注意力加权后的用户兴趣表示 [batch_size, embedding_dim]
            attention_weights: 注意力权重 [batch_size, max_seq_len]
        """
        batch_size, max_seq_len = keys.size(0), keys.size(1)
        
        # 扩展queries以匹配keys的维度
        queries = queries.unsqueeze(1).expand(-1, max_seq_len, -1)  # [batch_size, max_seq_len, embedding_dim]
        
        # 计算注意力得分
        queries_hidden = self.w_query(queries)  # [batch_size, max_seq_len, attention_size]
        keys_hidden = self.w_key(keys)  # [batch_size, max_seq_len, attention_size]
        
        # 激活
        hidden = F.relu(queries_hidden + keys_hidden)  # [batch_size, max_seq_len, attention_size]
        
        # 计算注意力权重
        attention_scores = self.w_value(hidden).squeeze(-1)  # [batch_size, max_seq_len]
        
        # 创建掩码，将无效的历史物品位置设为极小值
        mask = torch.arange(max_seq_len, device=keys_length.device).unsqueeze(0).expand(batch_size, -1)
        mask = mask < keys_length.unsqueeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e4)
        
        # 应用softmax得到权重
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, max_seq_len]
        
        # 用权重对历史物品嵌入向量加权求和
        output = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)  # [batch_size, embedding_dim]
        
        return output, attention_weights

class DIN(nn.Module):
    """
    Deep Interest Network模型实现
    使用注意力机制捕获用户的动态兴趣
    """
    def __init__(self, item_feat_dim, embedding_dim=64, attention_dim=64, mlp_hidden_dims=[128, 64, 32], dropout=0.2):
        super(DIN, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 物品嵌入层
        self.item_embedding = nn.Linear(item_feat_dim, embedding_dim)
        
        # 注意力层
        self.attention = Attention(embedding_dim, attention_dim)
        
        # MLP层
        mlp_layers = []
        input_dim = embedding_dim * 3  # 用户兴趣向量 + 候选物品向量 + 历史物品序列平均
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_layer = nn.Linear(mlp_hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, candidate_features, history_features, history_length):
        """
        前向传播函数
        
        参数:
            candidate_features: 候选物品特征 [batch_size, item_feat_dim]
            history_features: 历史物品特征序列 [batch_size, max_seq_len, item_feat_dim]
            history_length: 每个用户的历史序列实际长度 [batch_size]
        
        返回:
            logits: 预测得分 [batch_size, 1]
        """
        batch_size, max_seq_len = history_features.size(0), history_features.size(1)
        
        # 将原始特征转换为嵌入向量
        candidate_emb = self.item_embedding(candidate_features)  # [batch_size, embedding_dim]
        
        # 处理历史物品特征
        history_emb = self.item_embedding(history_features.view(-1, history_features.size(-1)))
        history_emb = history_emb.view(batch_size, max_seq_len, -1)  # [batch_size, max_seq_len, embedding_dim]
        
        # 计算注意力加权的用户兴趣表示
        interest_emb, attention_weights = self.attention(candidate_emb, history_emb, history_length)
        
        # 计算历史物品的平均向量作为额外特征
        mask = torch.arange(max_seq_len, device=history_length.device).unsqueeze(0) < history_length.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.embedding_dim).float()
        avg_history_emb = (history_emb * mask).sum(dim=1) / history_length.unsqueeze(-1).float()
        
        # 拼接特征
        concat_features = torch.cat([interest_emb, candidate_emb, avg_history_emb], dim=1)
        
        # 通过MLP层
        mlp_output = self.mlp(concat_features)
        
        # 输出层
        logits = self.output_layer(mlp_output)
        
        return self.sigmoid(logits).squeeze(-1)
    
    def calculate_loss(self, candidate_features, history_features, history_length, labels):
        """
        计算二分类交叉熵损失
        
        参数:
            candidate_features: 候选物品特征 [batch_size, item_feat_dim]
            history_features: 历史物品特征序列 [batch_size, max_seq_len, item_feat_dim]
            history_length: 每个用户的历史序列实际长度 [batch_size]
            labels: 真实标签 [batch_size]
            
        返回:
            loss: 交叉熵损失
        """
        pred = self.forward(candidate_features, history_features, history_length)
        loss = F.binary_cross_entropy(pred, labels.float())
        return loss 