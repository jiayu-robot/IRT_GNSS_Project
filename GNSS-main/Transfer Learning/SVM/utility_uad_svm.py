import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import config # 假设 config.py 在同一目录下
import os
def load_data(src_ids, tgt_id, csv_paths):#这个因为考虑越界的问题更改一下 原来的再下面被注释掉了
    # 只加载目标域数据，源域数据返回 None (因为会在序列制作时动态加载)
    print(f"[load_data] 正在加载目标域数据: {tgt_id}")
    df_tgt = pd.read_csv(csv_paths[tgt_id])
    
    print(f"[load_data] 目标域加载完成。形状: {df_tgt.shape}")
    print(f"[load_data] 注：源域数据将在 'create_all_sequences' 中逐个加载以保证时序独立性。")
    
    return None, df_tgt  # 第一个返回值给 None
# def load_data(src_ids, tgt_id, csv_paths):#加载源域和目标域数据
    
#     #目标域数据
#     df_tgt = pd.read_csv(csv_paths[tgt_id])
#     #源域的数据
#     df_sim_list = []
#     for i in src_ids:
#         df_sim = pd.read_csv(csv_paths[i])
#         df_sim_list.append(df_sim)

#     df_src = pd.concat(df_sim_list, ignore_index=True)
#     print(f"[load_data] 源域数据形状: {df_src.shape}, 目标域数据形状: {df_tgt.shape}")
#     return df_src, df_tgt
# =============================================================================
def make_sequences(df, features, target=None, seq_len=config.SEQ_LEN, step=config.STEP, 
                   group_col=config.GROUP_COL, time_col=config.TIME_COL):
    required_cols = features + [group_col, time_col]
    if target is not None:
        required_cols.append(target)
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"错误: 必需的列 '{col}' 在 DataFrame 中未找到。")
    df = df.copy()
    df = df.sort_values([group_col, time_col])
    X_list = []
    y_list = []
    grouped = df.groupby(group_col)
    for group_id, group_df in grouped:
        if len(group_df) < seq_len:
            continue
        arr_features = group_df[features].to_numpy(dtype=np.float32)
        arr_labels = None
        if target is not None:
            arr_labels = group_df[target].to_numpy(dtype=np.int64)
        # --- 4. 在该组内部执行滑动窗口 ---
        for i_start in range(0, len(arr_features) - seq_len + 1, step):
            i_end = i_start + seq_len
            X_list.append(arr_features[i_start:i_end])
            if arr_labels is not None:
                # 取序列中最后一个时间点的标签作为该序列的标签
                y_list.append(arr_labels[i_end - 1])
    if not X_list:
        print("[make_sequences] 警告: 没有生成任何序列。")
        # (返回空数组)
        empty_X = np.empty((0, seq_len, len(features)), dtype=np.float32)
        if target is not None:
            empty_y = np.empty((0,), dtype=np.int64)
            return empty_X, empty_y
        else:
            return empty_X
            
    X_out = np.asarray(X_list, dtype=np.float32)
    
    if target is not None:
        y_out = np.asarray(y_list, dtype=np.int64)
        print(f"  - X 形状: {X_out.shape}, y 形状: {y_out.shape}")
        return X_out, y_out
    else:
        print(f"  - X 形状: {X_out.shape}")
        return X_out
# =============================================================================   
import numpy as np
# 假设 config 模块已经导入
# from config import * # 假设 make_sequences 已经定义
def create_all_sequences(df_src_ignored, df_tgt, config):#新的序列化函数    把仿真数据逐个文件处理，防止边界跨越！！！！！！
    """
    [修改版] 逐个文件处理序列，防止边界跨越。
    
    参数:
    - df_src_ignored: 为了兼容旧接口保留的参数，但在函数内不使用它。
      (因为我们要重新从文件读取，以确保隔离)
    - df_tgt: 目标域数据 (通常是一个文件，直接传进来即可)
    - config: 配置对象
    """
    
    print("\n--- [步骤 2 - 修正版] 正在创建序列 (逐个Case独立处理)... ---")
    
    # ==========================================
    # 1. 处理源域 (仿真数据) - 逐个 Case 制作序列
    # ==========================================
    all_X_list = []
    all_y_list = []
    
    # 循环读取 config 中定义的每个源域 ID
    for src_id in config.SRC_IDS:
        csv_path = config.CSV_PATHS[src_id]
        print(f"  - 正在处理源域 Case {src_id}: {os.path.basename(csv_path)}")
        
        # 1. 读取单个 Case 文件
        df_case = pd.read_csv(csv_path)
        
        # 2. 马上制作该 Case 的序列
        # 因为只传入了当前 Case 的数据，滑动窗口绝对不会滑到别的 Case 去
        X_case, y_case = make_sequences(
            df=df_case, 
            features=config.FEATURES, 
            target=config.TARGET_COL,
            seq_len=config.SEQ_LEN, 
            step=config.STEP, 
            group_col=config.GROUP_COL, 
            time_col=config.TIME_COL
        )
        
        # 3. 收集结果
        if len(X_case) > 0:
            all_X_list.append(X_case)
            all_y_list.append(y_case)
        else:
            print(f"    [警告] Case {src_id} 没有生成有效序列，跳过。")

    # ==========================================
    # 2. 融合 (Merge)
    # ==========================================
    print("  - 正在合并所有源域序列...")
    if all_X_list:
        # 这里只是把做好的积木堆在一起，不会破坏积木内部的结构
        X_src = np.concatenate(all_X_list, axis=0)
        y_src = np.concatenate(all_y_list, axis=0)
        print(f"  - 源域合并完成。总序列数: {X_src.shape[0]}")
    else:
        print("Error: 没有生成任何源域序列！")
        return None, None, None

    # ==========================================
    # 3. 处理目标域 (真实数据)
    # ==========================================
    print("\n  - 正在处理目标域 (真实) 数据...")
    # 目标域通常是一个文件，直接处理即可
    X_tgt = make_sequences(
        df=df_tgt, 
        features=config.FEATURES, 
        target=None, # 无标签
        seq_len=config.SEQ_LEN, 
        step=config.STEP, 
        group_col=config.GROUP_COL, 
        time_col=config.TIME_COL
    )
    
    print("\n--- 序列创建完毕 ---")
    
    if X_src is None or X_tgt is None:
        print("错误: X_src 或 X_tgt 中没有生成任何序列。")
        return None, None, None
        
    return X_src, y_src, X_tgt
# def create_all_sequences(df_src, df_tgt, config):
#     """
#     [辅助包装函数]
#     调用 make_sequences 来转换源域和目标域的数据。

#     返回:
#     - X_src (np.ndarray): 仿真特征序列
#     - y_src (np.ndarray): 仿真标签序列
#     - X_tgt (np.ndarray): 真实特征序列 (无标签)
#     """
    
#     print("\n--- [步骤 2] 正在创建序列... ---")
    
#     # --- 2a. 处理源域 (仿真数据) ---
#     print("  - 正在处理源域 (仿真) 数据...")
#     # (!! 注意：这里传入了 target=config.TARGET_COL !!)
#     X_src, y_src = make_sequences(
#         df=df_src, 
#         features=config.FEATURES, 
#         target=config.TARGET_COL,
#         seq_len=config.SEQ_LEN, 
#         step=config.STEP, 
#         group_col=config.GROUP_COL, 
#         time_col=config.TIME_COL
#     )

#     # --- 2b. 处理目标域 (真实数据) ---
#     print("\n  - 正在处理目标域 (真实) 数据...")
#     # (!! 注意：这里 target=None !!)
#     X_tgt = make_sequences(
#         df=df_tgt, 
#         features=config.FEATURES, 
#         target=None, # <-- 关键：无标签
#         seq_len=config.SEQ_LEN, 
#         step=config.STEP, 
#         group_col=config.GROUP_COL, 
#         time_col=config.TIME_COL
#     )
    
#     print("\n--- 序列创建完毕 ---")
    
#     if X_src is None or X_tgt is None:
#         print("错误: X_src 或 X_tgt 中没有生成任何序列。")
#         return None, None, None
        
#     return X_src, y_src, X_tgt

# # [下一步] 回到 main() 函数中，用正确的函数名调用它。


# =============================================================================


class SeqDataset(Dataset):
    """
    模块4: 自定义 PyTorch Dataset
    """
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = y
        self.y_exists = y is not None
        if self.y_exists:
            self.y = torch.from_numpy(y).long()
        else:
            # 如果是目标域 (y=None), 创建一个占位符
            self.y = torch.zeros(len(self.X), dtype=torch.long) 
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    

    # =============================================================================
# 模块 5: PyTorch Dataloader (!! 你需要的部分 !!)
# =============================================================================
def create_dataloaders(X_src, y_src, X_tgt, batch_size=config.BATCH_SIZE):
    """
    [函数定义]
    使用 SeqDataset 和 DataLoader 来创建源域和目标域的数据加载器。
    
    返回:
    - dl_src (DataLoader): 源域(仿真)的“投喂器”
    - dl_tgt (DataLoader): 目标域(真实)的“投喂器”
    """
    
    print("\n--- [DataLoaders] 正在创建 DataLoaders... ---")
    
    # --- 3a. 创建源域 (仿真) "仓库" 和 "投喂器" ---
    # [理论] (1) 创建“仓库”
    ds_src = SeqDataset(X_src, y_src)
    
    # [理论] (2) 给“仓库”装上“投喂器”
    dl_src = DataLoader(
        ds_src,
        batch_size=batch_size,
        shuffle=True,       # [理论] 训练时必须打乱 (shuffle=True)
        drop_last=True      # 丢弃最后一个不完整的 batch
    )
    print(f" f - 源域 (仿真) 加载器: {len(ds_src)} 个样本, {len(dl_src)} 个批次。")

    # --- 3b. 创建目标域 (真实) "仓库" 和 "投喂器" ---
    # [理论] (1) 创建“仓库”，注意 y=None
    ds_tgt = SeqDataset(X_tgt, y=None)
    
    # [理论] (2) 给“仓库”装上“投喂器”
    dl_tgt = DataLoader(
        ds_tgt,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    print(f" f - 目标域 (真实) 加载器: {len(ds_tgt)} 个样本, {len(dl_tgt)} 个批次。")
    
    # --- 3c. 检查加载器是否为空 ---
    if len(dl_src) == 0 or len(dl_tgt) == 0:
        print("错误: 数据加载器为空。这可能是因为 BATCH_SIZE 大于样本数。")
        return None, None
        
    return dl_src, dl_tgt




#=============================================================================
# 给svm训练做准备
#=============================================================================
# -----------------------------------------------------------------------------
# 专为 SVM 平衡设计的新序列化函数
# -----------------------------------------------------------------------------

def make_sequences_for_svm(df, features, target=None, seq_len=config.SEQ_LEN, step=config.STEP, 
                           group_col=config.GROUP_COL, time_col=config.TIME_COL, sat_type_col=config.SAT_TYPE_COL):
    """
    制作序列，并额外返回卫星类型 S (Sat_Type)。
    """
    # 检查所有必需的列，包括卫星类型列 (略过检查代码...)
    required_cols = features + [group_col, time_col, sat_type_col]
    if target is not None:
        required_cols.append(target)
        
    for col in required_cols:
        if col not in df.columns:
            print(f"【错误】 make_sequences_for_svm 必需的列 '{col}' 未找到。")
            return None, None, None # 返回 None, None, None 确保流程中断
            
    df = df.sort_values([group_col, time_col])
    X_list = []
    y_list = []
    S_list = [] # <-- 卫星类型列表
    I_list = []
    
    grouped = df.groupby(group_col)
    
    for group_id, group_df in grouped:
        if len(group_df) < seq_len:
            continue
        arr_indices = group_df.index.to_numpy()    
        arr_features = group_df[features].to_numpy(dtype=np.float32)
        arr_labels = None
        
        if target is not None:
            arr_labels = group_df[target].to_numpy(dtype=np.int64)
            
        arr_sat_types = group_df[sat_type_col].to_numpy(dtype=np.int64)
        
        # --- 滑动窗口 ---
        for i_start in range(0, len(arr_features) - seq_len + 1, step):
            i_end = i_start + seq_len
            X_list.append(arr_features[i_start:i_end])
            I_list.append(arr_indices[i_end - 1])
            # 取序列中最后一个时间点的标签和卫星类型
            if arr_labels is not None:
                y_list.append(arr_labels[i_end - 1])
            
            S_list.append(arr_sat_types[i_end - 1]) # <-- 提取序列末尾的 sat_type
            
    if not X_list:
        return np.empty((0, config.SEQ_LEN, len(features)), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
            
    X_out = np.asarray(X_list, dtype=np.float32)
    S_out = np.asarray(S_list, dtype=np.int64)
    y_out = np.asarray(y_list, dtype=np.int64)
    I_raw_out = np.asarray(I_list, dtype=np.int64)
    print(f"  - X 形状: {X_out.shape}, y 形状: {y_out.shape}, S 形状: {S_out.shape}")
    return X_out, y_out, S_out, I_raw_out # <-- 统一返回 X, y, S, I



#     # ... 后续检查 ...
# 在 utility_uad_svm.py 中替换原有的 create_sequences_for_svm

def create_sequences_for_svm(df_src_ignored, df_tgt, config):
    """
    [修改版] 专门为 SVM 准备序列 (逐个文件处理)
    返回: X_src, y_src, S_src, I_src_raw, X_tgt, y_tgt, S_tgt, TGT_INDICES
    """
    print("\n--- [序列制作] 正在创建 SVM 所需的序列 (逐个Case处理)... ---")
    
    # ==========================================
    # 1. 源域处理 (循环读取)
    # ==========================================
    X_list, y_list, S_list, I_list = [], [], [], []
    
    for src_id in config.SRC_IDS:
        path = config.CSV_PATHS[src_id]
        print(f"  - 处理源域 Case {src_id}: {os.path.basename(path)}")
        
        df_case = pd.read_csv(path)
        
        # 调用 make_sequences_for_svm (这个底层函数不用改)
        # 注意：这里需要 target 存在
        X, y, S, I = make_sequences_for_svm(
            df=df_case, 
            features=config.FEATURES, 
            target=config.TARGET_COL,
            seq_len=config.SEQ_LEN, 
            step=config.STEP
        )
        
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)
            S_list.append(S)
            I_list.append(I)
    
    # 合并
    if X_list:
        X_src = np.concatenate(X_list, axis=0)
        y_src = np.concatenate(y_list, axis=0)
        S_src = np.concatenate(S_list, axis=0)
        I_src_raw = np.concatenate(I_list, axis=0)
        print(f"  - 源域合并完成。总数: {len(X_src)}")
    else:
        return None, None, None, None, None, None, None, None

    # ==========================================
    # 2. 目标域处理
    # ==========================================
    print("\n  - 处理目标域 (真实) 数据...")
    X_tgt, y_tgt, S_tgt, TGT_INDICES = make_sequences_for_svm(
        df=df_tgt, 
        features=config.FEATURES, 
        target=None, # 无标签
        seq_len=config.SEQ_LEN, 
        step=config.STEP
    )
    
    return X_src, y_src, S_src, I_src_raw, X_tgt, y_tgt, S_tgt, TGT_INDICES
# def create_sequences_for_svm(df_src, df_tgt, config):
#     print("\n--- [序列制作] 正在创建 SVM 所需的序列... ---")
    
#     # 1. 源域 (仿真数据) - 提取 X_src, y_src, S_src (有标签)
#     # 调用 make_sequences_for_svm 时需要 target=config.TARGET_COL
#     print("  - 正在处理源域 (仿真) 数据...")
#     # 注意：make_sequences_for_svm 应该返回 X, y, S
#     X_src, y_src, S_src ,I_src_raw= make_sequences_for_svm(df=df_src, features=config.FEATURES, 
#                                                  target=config.TARGET_COL)
    
#     if X_src is None:
#         print("错误: 源域序列制作失败。")
#         return None, None, None, None, None, None

#     # 2. 目标域 (真实数据) - 仅提取 X_tgt, S_tgt (无标签)
#     # 核心修改：目标域必须传入 target=None
#     print("\n  - 正在处理目标域 (真实) 数据 (无标签处理)...")
#     # 假设 make_sequences_for_svm 在 target=None 时，也返回 X, y_placeholder, S
#     # 或者我们手动处理返回值：
    
#     # 我们需要修改 make_sequences_for_svm 的返回值，让它在 target=None 时返回 X, S
#     # 但是为了简化流程，我们让它像以前一样返回 X, y_placeholder, S

#     X_tgt, y_tgt_placeholder, S_tgt ,TGT_INDICES= make_sequences_for_svm(df=df_tgt, features=config.FEATURES, 
#                                                                target=None) # <-- 关键修改：target=None
    
#     # 既然目标域没有真实标签 y_tgt 用于评估，我们用占位符 y_tgt_placeholder 即可。
#     # 如果目标域有标签用于评估，你需要单独加载它。但我们按 UDA 原则，假设它没有。
    
#     if X_tgt is None:
#         print("错误: 目标域序列制作失败。")
#         return None, None, None, None, None, None
    
#     print("--- 序列制作完成 ---")
    
#     if len(X_src) == 0 or len(X_tgt) == 0:
#         print(f"警告: 生成的序列数为零。源域数量: {len(X_src)}, 目标域数量: {len(X_tgt)}。")
#         return None, None, None, None, None, None

#     # 最终返回时，y_tgt 必须是一个 NumPy 数组（哪怕是 0）
#     # 我们用 y_tgt_placeholder 来代表没有真实标签的 y_tgt
#     return X_src, y_src, S_src,I_src_raw, X_tgt, y_tgt_placeholder, S_tgt,TGT_INDICES
# =============================================================================
# train SVM  Feature extracting
# =============================================================================
# ----------------------------------------------------------------------
# 1. 特征提取函数
# ----------------------------------------------------------------------
from tqdm import tqdm
from collections import Counter
def get_features(model_Gf, X_numpy, batch_size, device):
    """
    使用训练好的 G_f (LSTM) 模型批量提取整个数据集的特征。
    """
    print("  - 正在执行特征提取...")
    # 1. 转换回 PyTorch Dataset/DataLoader (使用 SeqDataset，它应该在你的文件中)
    ds = SeqDataset(X_numpy, y=None) 
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    model_Gf.eval() # 切换到评估模式
    all_features = []
    
    with torch.no_grad(): # 禁用梯度计算
        for X_batch, _ in tqdm(dl, desc="  - 提取特征中"): # 忽略占位符标签
            X_batch = X_batch.to(device)
            features = model_Gf(X_batch)
            all_features.append(features.cpu().numpy())
            
    features_out = np.concatenate(all_features, axis=0)
    print(f"  - 特征提取完毕。输出形状: {features_out.shape}")
    return features_out
# ----------------------------------------------------------------------
# 2. 复杂的欠采样平衡函数
# def complex_balance_data(X_feat, y_labels, sat_types, random_state=42):
#     """
#     基于标签 (y_labels) 和卫星类型 (sat_types) 对特征进行欠采样。
#     """
#     print(f"  - 原始类别分布: {Counter(y_labels)}")
    
#     df_temp = pd.DataFrame({'feat_idx': range(len(X_feat)), 'multipath': y_labels, 'sat_type': sat_types})
    
#     # 1. 分组 (四组)
#     gps_los_idx  = df_temp[(df_temp["sat_type"] == 0) & (df_temp["multipath"] == 0)]['feat_idx'].values
#     gps_nlos_idx = df_temp[(df_temp["sat_type"] == 0) & (df_temp["multipath"] == 1)]['feat_idx'].values
#     pl_los_idx  = df_temp[(df_temp["sat_type"] == 1) & (df_temp["multipath"] == 0)]['feat_idx'].values
#     pl_nlos_idx = df_temp[(df_temp["sat_type"] == 1) & (df_temp["multipath"] == 1)]['feat_idx'].values
    
#     # 2. 确定最小组大小
#     min_gps = min(len(gps_los_idx), len(gps_nlos_idx))
#     min_pl = min(len(pl_los_idx), len(pl_nlos_idx))
    
#     # 3. 对索引进行欠采样
#     # 注意：这里使用 np.random.choice 代替 sklearn.utils.resample，以简化依赖和类型处理
#     np.random.seed(random_state)
#     gps_los_bal_idx = np.random.choice(gps_los_idx, min_gps, replace=False)
#     gps_nlos_bal_idx = np.random.choice(gps_nlos_idx, min_gps, replace=False)
#     pl_los_bal_idx = np.random.choice(pl_los_idx, min_pl, replace=False)
#     pl_nlos_bal_idx = np.random.choice(pl_nlos_idx, min_pl, replace=False)

#     # 4. 合并索引并打乱
#     combined_idx = np.concatenate([gps_los_bal_idx, gps_nlos_bal_idx, pl_los_bal_idx, pl_nlos_bal_idx])
#     np.random.shuffle(combined_idx)
    
#     # 5. 重新映射到特征和标签
#     X_resampled = X_feat[combined_idx]
#     y_resampled = y_labels[combined_idx]

#     print("✅ 特征空间精细平衡完成。")
#     print(f" - GPS LOS/NLOS 各: {min_gps} 条")
#     print(f" - PL LOS/NLOS 各: {min_pl} 条")
#     print(f" - 平衡后总样本数: {len(X_resampled)}")
    
#     return X_resampled, y_resampled

def complex_balance_data(X_feat, y_labels, sat_types, random_state=42):
    """
    四象限平衡：GNSS好 / GNSS坏 / PL好 / PL坏 -> 数量强制相等
    """
    print(f"  - [平衡前] 总体分布: {Counter(y_labels)}")
    
    # 1. 找到四组的索引
    # 0=GNSS, 1=PL; 0=LOS, 1=Multipath
    idx_g_good = np.where((sat_types==0) & (y_labels==0))[0]
    idx_g_bad  = np.where((sat_types==0) & (y_labels==1))[0]
    idx_p_good = np.where((sat_types==1) & (y_labels==0))[0]
    idx_p_bad  = np.where((sat_types==1) & (y_labels==1))[0]
    
    print(f"    GNSS好: {len(idx_g_good)}, GNSS坏: {len(idx_g_bad)}")
    print(f"    PL好:   {len(idx_p_good)}, PL坏:   {len(idx_p_bad)}")
    
    # 2. 找最小值 (短板)
    counts = [len(idx_g_good), len(idx_g_bad), len(idx_p_good), len(idx_p_bad)]
    # 如果某一类完全没有数据，需要处理防止报错
    valid_counts = [c for c in counts if c > 0]
    if not valid_counts:
        print("❌ 严重错误：某类数据完全缺失，无法平衡！")
        return X_feat, y_labels
        
    min_count = min(valid_counts)
    print(f"  - 平衡基准数: {min_count} (每组取这么多)")
    
    # 3. 采样
    np.random.seed(random_state)
    # 辅助函数：如果该组有数据则采样，没数据则返回空
    def safe_sample(indices, n):
        if len(indices) == 0: return indices
        return np.random.choice(indices, n, replace=False)

    final_idx = np.concatenate([
        safe_sample(idx_g_good, min_count),
        safe_sample(idx_g_bad, min_count),
        safe_sample(idx_p_good, min_count),
        safe_sample(idx_p_bad, min_count)
    ])
    
    np.random.shuffle(final_idx)
    
    print(f"✅ 平衡完成。总样本数: {len(final_idx)}")
    return X_feat[final_idx], y_labels[final_idx]


def standardize_features(X_src_feat, X_tgt_feat):
    """
    对 LSTM 提取出的特征进行标准化（Standardization），并解决标准差为零的问题。

    参数:
    - X_src_feat (np.ndarray): 源域特征 (用于训练 Scaler)
    - X_tgt_feat (np.ndarray): 目标域特征 (用于应用 Scaler)

    返回:
    - X_src_feat_scaled (np.ndarray): 标准化后的源域特征
    - X_tgt_feat_scaled (np.ndarray): 标准化后的目标域特征
    """
    print("\n--- [标准化] 正在对特征进行标准化... ---")
    
    # 1. 训练 StandardScaler (仅在源域特征上)
    scaler = StandardScaler()
    scaler.fit(X_src_feat)
    
    # ----------------------------------------------------
    # 2. 关键修正：防御标准差为零 (sigma = 0)
    # ----------------------------------------------------
    
    # 找出所有标准差为 0 的维度
    zero_std_indices = scaler.scale_ == 0
    zero_std_count = np.sum(zero_std_indices)
    
    if zero_std_count > 0:
        # 将这些零标准差替换为一个极小的数 (epsilon=1e-8)
        scaler.scale_[zero_std_indices] = 1e-8
        print(f"【修复】成功修复 {zero_std_count} 个标准差为零的特征维度。")
    
    # ----------------------------------------------------
    # 3. 应用 Scaler 到源域和目标域
    # ----------------------------------------------------
    
    X_src_feat_scaled = scaler.transform(X_src_feat)
    X_tgt_feat_scaled = scaler.transform(X_tgt_feat)

    print("  - 特征标准化完成 (基于源域统计量)。")
    return X_src_feat_scaled, X_tgt_feat_scaled

# 假设 standardize_features 函数已在上方定义