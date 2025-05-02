# -*- coding: utf-8 -*-
# @Author  : 
# @Desc    : 集成奖励处理器

from typing import Dict, List, Any, Tuple, Callable
import numpy as np
import pandas as pd
from loguru import logger


class EnsembleProcessor:
    """集成奖励处理器，支持多种集成策略"""
    
    def __init__(self):
        """初始化集成处理器"""
        # 注册集成策略
        self.strategies = {
            'avg': self._strategy_average,
            'weighted': self._strategy_weighted,
            'voting': self._strategy_voting,
            'bon': self._strategy_best_of_n,
            'max_reward': self._strategy_max_reward,
            'min_rank': self._strategy_min_rank
        }
        
        # 配置要使用的集成策略
        self.active_ensembles = [
            # {'id': '1', 'name': 'Average', 'strategy': 'avg', 'params': {}},
            {'id': '2', 'name': 'Weighted', 'strategy': 'weighted',
             'params': {'weights': {'reward1': 0.2, 'reward3': 0.1, 'reward4': 0.6, 'reward5': 0.1}}},  # 'params': {'weights': {'reward4': 0.6, 'reward5': 0.5}}},
            # {'id': '3', 'name': 'Voting', 'strategy': 'voting', 'params': {"reward_names": ["reward1", "reward2", "reward3","reward4","reward5"]}},
            # {'id': '4', 'name': 'BoN', 'strategy': 'bon', 'params': {'n': 5}},
            # {'id': '5', 'name': 'MaxReward', 'strategy': 'max_reward', 'params': {}}
        ]
    
    def process_ensemble_rewards(self, result_df: pd.DataFrame, group: pd.DataFrame, 
                               sample_info: Dict[str, Any]) -> pd.DataFrame:
        """
        处理集成奖励并添加到结果DataFrame
        
        参数:
            result_df: 结果DataFrame
            group: 数据组
            sample_info: 样本分析信息
            
        返回:
            更新后的DataFrame
        """
        # 为每个激活的集成策略计算结果
        for ensemble in self.active_ensembles:
            ensemble_id = ensemble['id']
            strategy_name = ensemble['strategy']
            params = ensemble['params']
            
            # 获取策略函数
            strategy_func = self.strategies.get(strategy_name)
            if not strategy_func:
                logger.warning(f"未知的集成策略: {strategy_name}")
                continue
            
            # 计算集成奖励
            try:
                ensemble_reward = strategy_func(result_df, **params)
                result_df[f'ensemble{ensemble_id}'] = ensemble_reward
                
                # 找出最佳索引
                best_idx = ensemble_reward.idxmax()
                
                # 标记最佳选择
                result_df[f'is_ensemble{ensemble_id}_best'] = False
                result_df.loc[best_idx, f'is_ensemble{ensemble_id}_best'] = True
                
                # 添加排名
                result_df[f'ensemble{ensemble_id}_rank'] = result_df[f'ensemble{ensemble_id}'].rank(
                    ascending=False, method='min')
                
                # 记录到样本信息
                sample_info[f'ensemble{ensemble_id}_best_idx'] = best_idx
                sample_info[f'ensemble_score_{ensemble_id}'] = group.loc[best_idx, 'human_score']
                
                logger.debug(f"已计算集成策略 {ensemble['name']} (ID: {ensemble_id})")
            except Exception as e:
                logger.error(f"计算集成策略 {ensemble['name']} (ID: {ensemble_id}) 时出错: {str(e)}")
        
        return result_df
    
    def _strategy_average(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        集成策略: 平均所有奖励
        
        参数:
            df: 包含各个奖励的DataFrame
            
        返回:
            集成奖励值
        """
        # 获取所有reward列
        reward_cols = [col for col in df.columns if col.startswith('reward') 
                      and not col.endswith('_rank') and not col.endswith('_score') 
                      and not col.endswith('_best')]
        
        # 如果没有找到reward列，返回零序列
        if not reward_cols:
            return pd.Series(0, index=df.index)
        
        # 计算平均值
        return df[reward_cols].mean(axis=1)
    
    def _strategy_weighted(self, df: pd.DataFrame, weights: Dict[str, float] = None, **kwargs) -> pd.Series:
        """
        集成策略: 加权平均奖励
        
        参数:
            df: 包含各个奖励的DataFrame
            weights: 权重字典，键为奖励列名，值为权重
            
        返回:
            集成奖励值
        """
        if not weights:
            # 默认权重
            weights = {'reward1': 0.5, 'reward2': 0.3, 'reward3': 0.2}
        
        # 初始化结果
        result = pd.Series(0, index=df.index)
        
        # 计算加权和
        total_weight = 0
        for col, weight in weights.items():
            if col in df.columns:
                result += df[col] * weight
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            result = result / total_weight
        
        return result
    
    def _strategy_voting(self, df: pd.DataFrame, reward_names: List[str] = None, **kwargs) -> pd.Series:
        """
        集成策略: 基于排名一致性的投票机制
        对于每个解决方案，计算在不同reward中获得相同排名的次数
        排名越一致且排名越高，最终得分越高
        
        参数:
            df: 包含各个奖励的DataFrame
            reward_names: 指定参与投票的reward名称列表，如 ['reward1', 'reward2']
                        如果为None，则使用所有可用的reward
            
        返回:
            集成奖励值
        """
        # 获取要处理的reward列
        if reward_names:
            # 验证指定的reward是否存在
            reward_cols = [col for col in reward_names if col in df.columns]
            if not reward_cols:
                logger.warning(f"指定的reward列 {reward_names} 在数据中未找到，将使用所有可用的reward")
                reward_cols = [col for col in df.columns if col.startswith('reward') 
                             and not col.endswith('_rank') and not col.endswith('_score') 
                             and not col.endswith('_best')]
        else:
            # 使用所有可用的reward列
            reward_cols = [col for col in df.columns if col.startswith('reward') 
                         and not col.endswith('_rank') and not col.endswith('_score') 
                         and not col.endswith('_best')]
        
        if not reward_cols:
            return pd.Series(0, index=df.index)
        
        # 存储每个reward的排名
        ranks = pd.DataFrame(index=df.index)
        for col in reward_cols:
            ranks[col] = df[col].rank(ascending=False, method='min')
        
        # 计算每个位置的排名一致性得分
        consistency_scores = pd.Series(0.0, index=df.index)
        n_rewards = len(reward_cols)
        
        for idx in df.index:
            # 获取当前样本在各个reward中的排名
            current_ranks = ranks.loc[idx]
            
            # 计算每个排名出现的次数
            rank_counts = current_ranks.value_counts()
            
            # 计算一致性得分：考虑排名的位置和一致性
            score = 0
            for rank, count in rank_counts.items():
                # rank越小越好，count越大越好
                # 将rank归一化到[0,1]区间，1代表最好的排名
                normalized_rank = 1 - (rank - 1) / len(df)
                # 计算该排名的得分：排名越高、一致性越好，得分越高
                rank_score = normalized_rank * (count / n_rewards) ** 2
                score += rank_score
            
            consistency_scores[idx] = score
        
        # 归一化最终得分到[0,1]区间
        if consistency_scores.max() > consistency_scores.min():
            consistency_scores = (consistency_scores - consistency_scores.min()) / (consistency_scores.max() - consistency_scores.min())
        
        return consistency_scores
    
    def _strategy_best_of_n(self, df: pd.DataFrame, n: int = 3, **kwargs) -> pd.Series:
        """
        集成策略: Best-of-N
        选择每个样本在N个奖励中的最高分
        
        参数:
            df: 包含各个奖励的DataFrame
            n: 考虑的奖励数量
            
        返回:
            集成奖励值
        """
        # 获取所有reward列
        reward_cols = [col for col in df.columns if col.startswith('reward') 
                      and not col.endswith('_rank') and not col.endswith('_score') 
                      and not col.endswith('_best')]
        
        # 如果没有找到reward列，返回零序列
        if not reward_cols:
            return pd.Series(0, index=df.index)
        
        # 限制为前N个奖励
        reward_cols = reward_cols[:min(n, len(reward_cols))]
        
        # 对每个样本，取N个奖励中的最大值
        return df[reward_cols].max(axis=1)
    
    def _strategy_max_reward(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        集成策略: 最大奖励
        选择每个样本在所有奖励中的最高分
        
        参数:
            df: 包含各个奖励的DataFrame
            
        返回:
            集成奖励值
        """
        # 获取所有reward列
        reward_cols = [col for col in df.columns if col.startswith('reward') 
                      and not col.endswith('_rank') and not col.endswith('_score') 
                      and not col.endswith('_best')]
        
        # 如果没有找到reward列，返回零序列
        if not reward_cols:
            return pd.Series(0, index=df.index)
        
        # 对每个样本，取所有奖励中的最大值
        return df[reward_cols].max(axis=1)
    
    def _strategy_min_rank(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """
        集成策略: 最小排名
        基于每个样本在各个奖励中的最佳排名
        
        参数:
            df: 包含各个奖励的DataFrame
            
        返回:
            集成奖励值
        """
        # 获取所有rank列
        rank_cols = [col for col in df.columns if col.endswith('_rank') 
                    and col.startswith('reward')]
        
        # 如果没有找到rank列，返回零序列
        if not rank_cols:
            return pd.Series(0, index=df.index)
        
        # 对每个样本，取所有排名中的最小值（排名越小越好）
        min_ranks = df[rank_cols].min(axis=1)
        
        # 返回排名的倒数（使得排名越小，得分越高）
        return 1 / min_ranks
    
    def analyze_ensemble_rewards(self, group: pd.DataFrame, best_human_score: float, 
                               top_k_scores: List[float]) -> Dict[str, Any]:
        """
        分析集成奖励的效果
        
        参数:
            group: 数据组
            best_human_score: 最佳人类评分
            top_k_scores: 前k个最佳人类评分
            
        返回:
            分析结果字典
        """
        result_df = group.copy()
        results = {}
        
        # 为每个激活的集成策略计算结果
        for ensemble in self.active_ensembles:
            ensemble_id = ensemble['id']
            strategy_name = ensemble['strategy']
            params = ensemble['params']
            
            # 获取策略函数
            strategy_func = self.strategies.get(strategy_name)
            if not strategy_func:
                continue
            
            # 计算集成奖励
            try:
                ensemble_reward = strategy_func(result_df, **params)
                best_idx = ensemble_reward.idxmax()
                
                # 记录结果
                results[f'ensemble{ensemble_id}_best_idx'] = best_idx
                results[f'select_score_ensemble{ensemble_id}'] = group.loc[best_idx, 'human_score']
                results[f'ensemble{ensemble_id}_best1_is_correct'] = (
                    group.loc[best_idx, 'human_score'] == best_human_score)
                results[f'ensemble{ensemble_id}_best3_is_correct'] = (
                    group.loc[best_idx, 'human_score'] in top_k_scores)
            except Exception as e:
                logger.error(f"分析集成策略 {ensemble['name']} (ID: {ensemble_id}) 时出错: {str(e)}")
        
        return results
    
    def get_ensemble_numbers(self) -> List[str]:
        """
        获取激活的集成策略编号
        
        返回:
            编号列表
        """
        return [ensemble['id'] for ensemble in self.active_ensembles]
    
    def get_ensemble_headers(self) -> List[str]:
        """
        获取集成奖励的表头
        
        返回:
            表头列表
        """
        headers = []
        for ensemble in self.active_ensembles:
            headers.append(f"Ensemble{ensemble['id']}_{ensemble['name']}_Score")
        return headers
    
    def get_ensemble_scores(self, sample_info: Dict[str, Any]) -> List[float]:
        """
        获取集成奖励的分数
        
        参数:
            sample_info: 样本信息
            
        返回:
            分数列表
        """
        scores = []
        for ensemble in self.active_ensembles:
            ensemble_id = ensemble['id']
            scores.append(sample_info.get(f'ensemble_score_{ensemble_id}', 0))
        return scores
    
    def add_ensemble_strategy(self, strategy_id: str, name: str, strategy_func: Callable, params: Dict = None):
        """
        添加自定义集成策略
        
        参数:
            strategy_id: 策略ID
            name: 策略名称
            strategy_func: 策略函数
            params: 策略参数
        """
        # 注册策略函数
        strategy_name = f"custom_{strategy_id}"
        self.strategies[strategy_name] = strategy_func
        
        # 添加到激活的集成策略
        self.active_ensembles.append({
            'id': strategy_id,
            'name': name,
            'strategy': strategy_name,
            'params': params or {}
        })
        
        logger.info(f"已添加自定义集成策略: {name} (ID: {strategy_id})") 