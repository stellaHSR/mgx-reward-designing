# -*- coding: utf-8 -*-
# @Author  : 
# @Desc    :
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class RewardMetrics:
    """奖励指标数据类"""
    reward: Optional[float] = None
    best_idx: Optional[int] = None
    select_score: Optional[float] = None
    best1_correct: bool = False
    best3_correct: bool = False
    human_rank: Optional[float] = None
    
    def to_dict(self, prefix: str) -> Dict[str, Any]:
        """转换为字典格式，用于存储结果"""
        return {
            f"{prefix}_best_idx": self.best_idx,
            f"select_score_{prefix}": self.select_score,
            f"{prefix}_best1_is_correct": self.best1_correct,
            f"{prefix}_best3_is_correct": self.best3_correct,
            f"{prefix}_human_rank": self.human_rank
        }


@dataclass
class HumanScoreInfo:
    """人类评分信息数据类"""
    scores: List[float]
    best_score: float
    avg_score: float
    ranks: pd.Series
    top3_scores: List[float] = field(default_factory=list)
    
    @classmethod
    def from_scores(cls, scores: pd.Series) -> 'HumanScoreInfo':
        """从分数序列创建实例"""
        sorted_scores = scores.sort_values(ascending=False).values.tolist()
        return cls(
            scores=sorted_scores,
            best_score=scores.max(),
            avg_score=scores.mean(),
            ranks=scores.rank(ascending=False, method='min'),
            top3_scores=sorted_scores[:3]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "human_scores": self.scores,
            "best_human_score": self.best_score,
            "avg_human_score": self.avg_score,
            "top3_human_scores": self.top3_scores
        }


class RewardCalculator:
    """奖励计算器类，负责计算不同的奖励函数"""
    
    @staticmethod
    def calculate_reward1(group: pd.DataFrame) -> pd.Series:
        """
        计算奖励函数1
        基于可执行性、LLM评分、代码行数和代码文件数量
        """
        return group['executability'] * (
                0.6 * group['llm_score'] +
                0.2 * (group['code_lines'] / group['code_lines'].max()) +
                0.2 * (group['project_files_count'] / group['project_files_count'].max())
        )
    
    @staticmethod
    def calculate_reward2(group: pd.DataFrame) -> pd.Series:
        """
        计算奖励函数2
        基于可执行性、LLM评分、成本和代码行数的归一化值
        """
        max_cost = group['cost($)'].max()
        max_code_lines = group['code_lines'].max()
        
        return group['executability'] * (
                group['llm_score'] +
                (group['cost($)'] / max_cost if max_cost > 0 else 0) * 0.5 +
                (group['code_lines'] / max_code_lines if max_code_lines > 0 else 0) * 0.2
        )
    
    @staticmethod
    def calculate_reward3(group: pd.DataFrame) -> pd.Series:
        """
        计算奖励函数3
        主要基于可执行性和LLM评分
        """
        
        # 计算cost奖励/惩罚项
        cost_factor = pd.Series(index=group.index)
        cost_factor[group['cost($)'] <= 2] = 0.2 * group['cost($)']  # cost <= 2时为正比例
        cost_factor[group['cost($)'] > 2] = -0.2 * group['cost($)']  # cost > 2时为负比例
        
        return 1 * group['executability'] * (
                0.70 * group['llm_score'] +
                cost_factor  # 添加cost奖励/惩罚项
        )
    
    @staticmethod
    def calculate_reward4(group: pd.DataFrame) -> pd.Series:
        """
        计算奖励函数4
        基于可执行性、LLM评分和归一化的成本与代码行数
        """
        # 创建生产力指标，使用float64类型
        productivity = pd.Series(0.0, index=group.index, dtype='float64')  # 修改这里
        mask = group['total_time'] > 0
        productivity[mask] = group.loc[mask, 'project_files_count'] / group.loc[mask, 'total_time']
        
        return group['executability'] * productivity * (
                0.7 * group['llm_score']
        )
    
    @staticmethod
    def calculate_reward5(group: pd.DataFrame) -> pd.Series:
        """
        计算奖励函数5
        考虑时间效率和成本效益
        """
        # 创建效率指标，使用float64类型
        efficiency = pd.Series(0.0, index=group.index, dtype='float64')  # 修改这里
        mask = group['total_time'] > 0
        efficiency[mask] = (group.loc[mask, 'code_lines'] * group.loc[mask, 'project_files_count']) / (
                group.loc[mask, 'total_time'] * group.loc[mask, 'cost($)'])
        
        # 处理无穷大值
        efficiency = efficiency.replace([np.inf, -np.inf], 0.0)
        
        return group['executability'] * (
                0.7 * group['llm_score'] +
                0.3 * (efficiency / efficiency.max() if efficiency.max() > 0 else 0.0)
        )
