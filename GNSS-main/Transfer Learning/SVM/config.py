import os
import torch
import pandas as pd
# =============================================================================
# 1. 全局设置 (Global Settings)
# =============================================================================
# 检查是否有可用的 GPU (NVIDIA CUDA)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- [Config] 正在使用的设备: {DEVICE} ---")

# =============================================================================
# 2. 数据参数 (Data Parameters)
# =============================================================================
# --- 2a. 数据列名 ---
# (!! 这是你必须确保正确的地方 !!)

# 特征列表 
FEATURES = ["cn0", "elevation", "pseudorange_residual", "doppler_shift","sat_type"]#"doppler_shift",

# 你的标签列名 (在仿真数据中)
TARGET_COL = "multipath"


SAT_TYPE_COL = "sat_type"  # 你的卫星类型列名
# 你的卫星ID列名 (用于分组)
GROUP_COL = "sv_id"

# 你的时间排序列名 (用于排序)
TIME_COL = "gps_time" # (!! 如果你的时间列名不同，请修改这里 !!)


# --- 2b. 序列化参数 ---
# (!! 这是你必须确保正确的地方 !!)

# 序列长度 (对应你数据中的 5)
SEQ_LEN = 5

# 滑动窗口的步长 (Step)
# step=1 会产生最多的数据 (高度重叠)
# step=5 会产生无重叠的数据
STEP = 1  # (!! 你可以根据需要调整这个值 !!)


# --- 2c. 数据路径 ---
# (!! 这是你必须确保正确的地方 !!)

# # 获取项目根目录 (假设 config.py 在项目文件夹中)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#base_dir = os.getcwd()
#print("当前工作目录:", base_dir)
parent_dir = os.path.dirname(os.path.dirname(BASE_DIR))
CSV_PATHS = {0: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "observation_data_case5_with_sat_type.csv"),
            1: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "Case1_Urban_10Hz_with_sat_type.csv"),
            2: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "Case1_Suburban_10Hz_with_sat_type.csv"),
            3: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "Case2_Urban_10Hz_with_sat_type.csv"),
            4: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "Case2_Suburban_10Hz_with_sat_type.csv"),
            5: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "Case3_Urban_10Hz_with_sat_type.csv"),
            6: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "Case3_Suburban_10Hz_with_sat_type.csv"),
            7: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "Case4_Urban_10Hz_with_sat_type.csv"),
            8: os.path.join(parent_dir, "Transfer Learning\data_with_sat_type", "Case4_Suburban_10Hz_with_sat_type.csv"),
            }

SRC_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
TGT_ID = 0

MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")

# 确保保存模型的文件夹存在
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# # 目标域 (真实) 数据路径
# # (!! 请确保这个路径是正确的 !!)
# TGT_DATA_PATH = os.path.join(DATA_DIR, "RWD_20250422/D1T4", "observation_data_case5_1.csv")

# =============================================================================
# 3. 模型超参数 (Model Hyperparameters)
# =============================================================================
# --- 3a. G_f (LSTM 特征提取器) ---

# LSTM 输入特征数 (自动从 FEATURES 列表计算)
LSTM_INPUT_SIZE = len(FEATURES) # 这将是 5

# LSTM 隐藏层的大小 (这是你可以调整的关键参数)
LSTM_HIDDEN_SIZE = 128

# LSTM 的层数
LSTM_NUM_LAYERS = 2

# LSTM 的 Dropout 率 (仅在 num_layers > 1 时激活)
LSTM_DROPOUT = 0.5


# --- 3b. G_y (标签分类器) 和 G_d (域判别器) ---

# 分类器的中间隐藏层维度
CLASSIFIER_HIDDEN_DIM = 128

# 最终标签分类的数量
# (例如 0="好信号", 1="坏信号/多路径")
NUM_CLASSES = 2 


# =============================================================================
# 4. 训练超参数 (Training Hyperparameters)
# =============================================================================
# 批次大小 (Batch Size)
BATCH_SIZE = 64 # (这是你可以调整的关键参数)

# 训练轮数 (Epochs)
NUM_EPOCHS = 50 # (这是一个起始点, 你可能需要根据验证集表现来调整)

# 学习率 (Learning Rate)
# (这是你可以调整的最关键的参数)
LEARNING_RATE_G = 0.0001# G_f 和 G_y 的学习率
LEARNING_RATE_D = 0.00005 # G_d 的学习率 (有时设得小一点会更稳定)

# GRL 层的 Alpha 值 (用于控制对抗损失的强度)
GRL_ALPHA = 0.


