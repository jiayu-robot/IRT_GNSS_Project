import numpy as np
import pandas as pd
import pymap3d as pm # pip install pymap3d
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

def weighted_multi_constellation_spp(sat_positions, pseudoranges, sat_types, weights, x0=None):
    """
    - x0: 修正为地球表面
    - sat_types: 0=天基GNSS, 1=地基PL
    """
    
    if x0 is None: 
        # 必须从地球表面开始解算, 不能从地心[0,0,0]
        x0 = np.array([6371000.0, 0.0, 0.0, 0.0, 0.0])

    def weighted_residuals(x):
        # 解包 5 个未知数
        receiver_position, dt_gnss, dt_pl = x[:3], x[3], x[4]
        
        # 几何距离 (您的sat_positions已经是混合了Meters的GNSS和PL坐标)
        geom_dist = np.linalg.norm(sat_positions - receiver_position, axis=1)
        
        # 根据 sat_type 应用不同的钟差
        # 如果 sat_type == 0 (GNSS), 钟差 = dt_gnss
        # 如果 sat_type == 1 (PL), 钟差 = dt_pl
        clock_corrections = np.where(sat_types == 0, dt_gnss, dt_pl)
        
        # 预测伪距 = 几何距离 + 对应的钟差
        predicted_pseudoranges = geom_dist + clock_corrections
        
        residuals = pseudoranges - predicted_pseudoranges
        return weights * residuals

    # 求解
    result = least_squares(weighted_residuals, x0, method='lm')
    # 返回: 位置, GNSS钟差, PL钟差
    return result.x[:3], result.x[3], result.x[4]


def multi_constellation_spp(sat_positions, pseudoranges, sat_types, x0=None):
    """
    标准 SPP (5参数) - 最终版
    (它只是调用加权版本，但权重全部为1)
    """
    # 创建一个全为 1.0 的权重数组
    weights = np.ones(len(pseudoranges))
    
    # 直接调用您的加权函数
    return weighted_multi_constellation_spp(sat_positions, pseudoranges, sat_types, weights, x0)

print("✅ 最终版5参数混合解算器 (GNSS + PL) 已准备就绪。")