import torch
import torch.nn as nn
from torch.autograd import Function

# ------------------------------------------------------------------------------
# 1. 梯度反转层 (GRL) - UDA的核心
# ------------------------------------------------------------------------------

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL_Layer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL_Layer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradReverse.apply(x, self.alpha)

# ------------------------------------------------------------------------------
# 2. LSTM 特征提取器 (G_f)
# ------------------------------------------------------------------------------

class LSTMFeatureExtractor(nn.Module):
    """
    [修改版] 特征提取器 (G_f)
    - 包含了 fc_final 层 (BatchNormalization + LeakyReLU)
    - 这是解决 0 值问题和不稳定的核心！
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, final_feature_dim=128):
        # 注意：增加了 final_feature_dim 参数，建议设为 128 或 64
        super(LSTMFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,    
            hidden_size=hidden_size,  
            num_layers=num_layers,
            batch_first=True,         
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False 
        )
        
        # --- ！！！关键修改开始！！！ ---
        # 添加投影层：将 LSTM 特征映射到更鲁棒的空间，并进行归一化
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_size, final_feature_dim), # 映射
            nn.BatchNorm1d(final_feature_dim),         # 归一化 (防止特征过大或过小)
            nn.LeakyReLU(0.1)                          # 激活 (防止死 0)
        )
        # --- ！！！关键修改结束！！！ ---

    def forward(self, x):
        # x: (batch, 5, 4)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        output, _ = self.lstm(x, (h_0, c_0))
        
        # 取最后一个时间步
        raw_feature = output[:, -1, :] 
        
        # --- ！！！关键修改！！！ ---
        # 通过投影层
        final_feature = self.fc_final(raw_feature)
        
        return final_feature
# class LSTMFeatureExtractor(nn.Module):
#     """
#     特征提取器 (G_f)
#     - 接收 (batch, 5, 4) 的序列
#     - 输出 (batch, hidden_size) 的特征向量
#     """
#     def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
#         """
#         参数:
#         - input_size: 特征数 (在你的例子中是 4)
#         """
#         super(LSTMFeatureExtractor, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.lstm = nn.LSTM(
#             input_size=input_size,    
#             hidden_size=hidden_size,  
#             num_layers=num_layers,
#             batch_first=True,         
#             dropout=dropout if num_layers > 1 else 0,
#             #bidirectional=True  # <-- ！！[修改 1] 在这里添加！！
#         )
        
# #单向LSTM
#     def forward(self, x):
#         # x 形状: (batch_size, 5, 4)
#         h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # output 形状: (batch_size, 5, hidden_size)
#         output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
#         # 我们使用最后一个时间步的输出作为整个序列的“总结”特征
#         features = output[:, -1, :] # 形状: (batch_size, hidden_size)
        
#         return features
#双向LSTM
    # def forward(self, x):
    #     """
    #     [!! 修改后的 forward !!]
    #     """
    #     # x 形状: (batch_size, seq_len, input_size)
        
    #     # !! [修改 2] 双向网络需要 num_layers * 2 !!
    #     # h_0 和 c_0 是 LSTM 的初始隐藏状态和细胞状态
    #     h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
    #     c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
    #     # output 形状: (batch_size, seq_len, hidden_size * 2)
    #     # h_n 形状: (num_layers * 2, batch_size, hidden_size)
    #     output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
    #     # ----------------------------------------------------
    #     # ！！ [修改 3] 提取最终的“前向”和“后向”状态 ！！
    #     # ----------------------------------------------------
    #     # h_n[-2, :, :] 是最后一个“前向”层的最终状态 (t=seq_len)
    #     # h_n[-1, :, :] 是最后一个“后向”层的最终状态 (t=0)
        
    #     # 我们将它们拼接 (concatenate) 在一起
    #     features = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
    #     # 最终 features 形状: (batch_size, hidden_size * 2)
        
    #     return features
# ------------------------------------------------------------------------------
# 3. 标签分类器 (G_y)
# ------------------------------------------------------------------------------

class LabelClassifier(nn.Module):
    """
    标签分类器 (G_y)
    - 预测信号是 "好" 还是 "坏" (多路径)
    """
    def __init__(self, input_size, num_classes, hidden_dim=64):
        super(LabelClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim), # input_size 是 LSTM 的 hidden_size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes) # num_classes (例如 2, 代表"好"和"坏")
        )
        
    def forward(self, x):
        # x 形状: (batch_size, hidden_size)
        logits = self.network(x) # 形状: (batch_size, num_classes)
        return logits

# ------------------------------------------------------------------------------
# 4. 域判别器 (G_d)
# ------------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """
    域判别器 (G_d)
    - 预测特征来自 "仿真" 还是 "真实"
    """
    def __init__(self, input_size, hidden_dim=64):
        super(DomainDiscriminator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim), # input_size 是 LSTM 的 hidden_size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2) # 2个类别: 0 (仿真) vs 1 (真实)
        )
        
    def forward(self, x):
        # x 形状: (batch_size, hidden_size)
        logits = self.network(x) # 形状: (batch_size, 2)
        return logits