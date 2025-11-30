import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import least_squares
import os


def weighted_single_point_positioning(sat_positions, pseudoranges, weights, x0=None):
    """
    使用加权最小二乘法 (WLS) 进行单点定位。
    """
    if x0 is None:
        x0 = np.zeros(4)

    def weighted_residuals(x):
        receiver_position, clock_bias = x[:3], x[3]
        predicted_pseudoranges = np.linalg.norm(sat_positions - receiver_position, axis=1) + clock_bias
        residuals = pseudoranges - predicted_pseudoranges
        # 核心区别：返回加权后的残差
        return weights * residuals

    result = least_squares(weighted_residuals, x0, method='lm')
    return result.x[:3], result.x[3]

# print("新增的 'weighted_single_point_positioning' 函数定义完成。")




import numpy as np
from scipy.optimize import least_squares

import numpy as np
from scipy.optimize import least_squares

def weighted_multi_constellation_spp(sat_positions, pseudoranges, sat_types, weights, x0=None):
    """
    [修改版] 智能加权 SPP 解算器
    - 自动检测是否需要解算系统间偏差 (ISB)。
    - 如果 sat_types 只有一种数值 (例如全0)，只解 4 参数 (x,y,z,dt)。
    - 如果 sat_types 有多种数值，解 5 参数 (x,y,z,dt1,dt2)。
    """
    
    # 1. 检查有几种卫星系统
    unique_types = np.unique(sat_types)
    num_systems = len(unique_types)
    
    # 2. 确定未知数数量 (3个坐标 + N个钟差)
    # 如果 sat_types 全是 0，这里 num_systems 就是 1，总参数就是 4
    num_params = 3 + num_systems 
    
    # 3. 初始化初值 x0
    if x0 is None:
        # 地球半径作为起点，避免地心陷阱
        # 后面补 0 (对应钟差)
        x0 = np.concatenate(([6371000.0, 0.0, 0.0], np.zeros(num_systems)))
    else:
        # 如果传入了 x0，确保长度对齐 (比如上一历元的解是4维，这一历元需要5维)
        if len(x0) != num_params:
             x0 = np.concatenate(([6371000.0, 0.0, 0.0], np.zeros(num_systems)))

    def weighted_residuals(x):
        # 解包位置
        receiver_position = x[:3]
        
        # 几何距离
        geom_dist = np.linalg.norm(sat_positions - receiver_position, axis=1)
        
        # 处理钟差
        if num_systems == 1:
            # --- 单系统模式 (4参数) ---
            # 适用于：只有GPS，或者 PL/GPS 钟差已对齐并强制设为 type 0
            clock_bias = x[3]
            predicted_pseudoranges = geom_dist + clock_bias
            
        else:
            # --- 多系统模式 (5参数) ---
            # 适用于：同时传入了 type 0 和 type 1，且需要解 ISB
            # x[3] 对应 unique_types[0] (通常是 0/GNSS)
            # x[4] 对应 unique_types[1] (通常是 1/PL)
            
            clock_corrections = np.zeros(len(pseudoranges))
            for i, sys_id in enumerate(unique_types):
                # 找到属于该系统的卫星索引
                mask = (sat_types == sys_id)
                # 加上对应的钟差参数 (x[3+i])
                clock_corrections[mask] = x[3 + i]
                
            predicted_pseudoranges = geom_dist + clock_corrections
        
        residuals = pseudoranges - predicted_pseudoranges
        return weights * residuals

    # 4. 求解
    # 使用 Levenberg-Marquardt 算法 (method='lm') 求解非线性最小二乘
    result = least_squares(weighted_residuals, x0, method='lm')
    
    # 返回结果 (注意：为了兼容性，我们只返回坐标，钟差丢弃或按需返回)
    # 如果需要 debug 钟差，可以 print(result.x[3:])
    return result.x[:3], 0, 0 # 后两个返回值是占位符，保持接口一致


def multi_constellation_spp(sat_positions, pseudoranges, sat_types, x0=None):
    """
    标准 SPP (等权) 包装器
    """
    weights = np.ones(len(pseudoranges))
    return weighted_multi_constellation_spp(sat_positions, pseudoranges, sat_types, weights, x0)



#     # 创建一个全为 1.0 的权重数组
#     weights = np.ones(len(pseudoranges))
    
#     # 直接调用您的加权函数
#     return weighted_multi_constellation_spp(sat_positions, pseudoranges, sat_types, weights, x0)

# print("✅ 最终版5参数混合解算器 (GNSS + PL) 已准备就绪。")