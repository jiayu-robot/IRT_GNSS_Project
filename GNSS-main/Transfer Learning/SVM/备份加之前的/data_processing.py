import numpy as np
import pandas as pd
import config 

def make_sequences(df, features, target=None, seq_len=config.SEQ_LEN, step=config.STEP, 
                   group_col="sv_id", time_col="gps_time"):
    
    print(f"[make_sequences] 开始创建序列...")
    print(f"  - 序列长度 (seq_len): {seq_len}, 步长 (step): {step}")
    print(f"  - 特征 (features): {features}")
    print(f"  - 目标 (target): {'无 (目标域)' if target is None else target + ' (源域)'}")
    
    # --- 1. 验证输入 ---
    # 确保所有必需的列都存在于 DataFrame 中
    required_cols = features + [group_col, time_col]
    if target is not None:
        required_cols.append(target)
        
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"[make_sequences] 错误: 必需的列 '{col}' 在 DataFrame 中未找到。")
            
    # --- 2. 准备数据 ---
    # 创建一个副本以避免修改原始 DataFrame
    df = df.copy()
    
    # **关键步骤**: 按组和时间对整个 DataFrame 进行排序
    # 这确保了当我们按组迭代时，每个组内部的数据已经是按时间排好序的。
    print(f"  - 正在按 {group_col} 和 {time_col} 排序数据...")
    df = df.sort_values([group_col, time_col])

    # 准备用于存储所有序列的列表
    X_list = []
    y_list = []
    
    # --- 3. 按组迭代并创建序列 ---
    # 这是此函数的核心逻辑：我们对每个 "sv_id" 单独处理。
    print(f"  - 正在按 '{group_col}' 分组并处理...")
    
    grouped = df.groupby(group_col)
    total_groups = len(grouped)
    
    for i, (group_id, group_df) in enumerate(grouped):
        
        # 打印进度
        if (i+1) % 100 == 0 or (i+1) == total_groups:
             print(f"    - 正在处理组 {i+1}/{total_groups} (ID: {group_id})")

        # --- 3a. 检查该组的数据是否足够长 ---
        if len(group_df) < seq_len:
            # print(f"    - 跳过组 {group_id}: 数据长度 ({len(group_df)}) 小于 seq_len ({seq_len})。")
            continue # 跳过这个组

        # --- 3b. 将该组的特征和标签转换为 Numpy 数组 (性能更高) ---
        
        # **关键步骤**: 只提取 `features` 列表中的列。
        # `group_col` ("sv_id") 绝不会在这里被包含，因此它永远不会成为模型的特征。
        arr_features = group_df[features].to_numpy(dtype=np.float32)
        
        arr_labels = None
        if target is not None:
            arr_labels = group_df[target].to_numpy(dtype=np.int64)
            
        # --- 3c. 在该组内部执行滑动窗口 ---
        # `stop` 位置是 `len(arr_features) - seq_len + 1`
        # `range(start, stop, step)`
        for i_start in range(0, len(arr_features) - seq_len + 1, step):
            i_end = i_start + seq_len
            
            # 提取特征序列
            X_list.append(arr_features[i_start : i_end])
            
            # 如果有标签，提取对应的标签
            if arr_labels is not None:
                # 标签策略：使用窗口的最后一个时间步对应的标签
                y_list.append(arr_labels[i_end - 1])

    # --- 4. 检查是否生成了任何数据 ---
    if not X_list:
        print("[make_sequences] 警告: 没有生成任何序列。")
        print("  - 请检查你的数据长度、`seq_len` 和 `step` 参数。")
        
        # 返回正确形状的空数组，以防下游代码出错
        empty_X = np.empty((0, seq_len, len(features)), dtype=np.float32)
        if target is not None:
            empty_y = np.empty((0,), dtype=np.int64)
            return empty_X, empty_y
        else:
            return empty_X
            
    # --- 5. 转换为最终的 Numpy 数组并返回 ---
    X_out = np.asarray(X_list, dtype=np.float32)
    
    if target is not None:
        y_out = np.asarray(y_list, dtype=np.int64)
        print(f"[make_sequences] 创建完成。")
        print(f"  - X 形状: {X_out.shape}")
        print(f"  - y 形状: {y_out.shape}")
        return X_out, y_out
    else:
        print(f"[make_sequences] 创建完成 (无标签)。")
        print(f"  - X 形状: {X_out.shape}")
        return X_out