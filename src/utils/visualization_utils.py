import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from typing import Dict, List
from loguru import logger

def setup_chinese_font() -> FontProperties:
    """
    设置中文字体
    
    返回:
        FontProperties: 字体对象
    """
    try:
        font = FontProperties(fname=r'C:\Windows\Fonts\SimHei.ttf')
    except:
        try:
            font = FontProperties(fname=r'/usr/share/fonts/SimHei.ttf')
        except:
            font = FontProperties()
    return font

def plot_rewards_trend(
    trend_data: Dict[str, List],
    reward_numbers: List[str],
    sample_sizes: List[int],
    avg_human_scores: Dict[int, float],
    best_human_scores: Dict[int, float],
    save_path: str
) -> None:
    """绘制奖励趋势图"""
    plt.figure(figsize=(12, 8))
    
    # 扩展颜色和标记列表
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    markers = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '8', 
               '*', 'H', '+', 'x', 'd']
    
    # 确保颜色和标记列表足够长
    while len(colors) < len(set(trend_data['reward_method'])):
        colors.extend(colors)
    while len(markers) < len(set(trend_data['reward_method'])):
        markers.extend(markers)
    
    # 绘制每个reward方法的趋势线
    unique_methods = sorted(set(trend_data['reward_method']))
    for idx, method in enumerate(unique_methods):
        mask = [m == method for m in trend_data['reward_method']]
        x_values = [trend_data['sample_size'][i] for i in range(len(mask)) if mask[i]]
        y_values = [trend_data['mean_score'][i] for i in range(len(mask)) if mask[i]]

        plt.plot(
            x_values, y_values,
            label=method,
            color=colors[idx],
            marker=markers[idx],
            markersize=8,
            linewidth=2
        )
       
    
    # 绘制人类评分参考线
    plt.plot(sample_sizes, [avg_human_scores[size] for size in sample_sizes],
             'k--', label='Average Human Score', linewidth=2)
    plt.plot(sample_sizes, [best_human_scores[size] for size in sample_sizes],
             'k:', label='Best Human Score', linewidth=2)
    
    plt.xlabel('Sample Size')
    plt.ylabel('Score')
    plt.title('Reward Methods Performance Trend')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

