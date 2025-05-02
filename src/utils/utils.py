# -*- coding: utf-8 -*-
# @Author  : 
# @Desc    : 工具函数模块

from typing import  List, Tuple
import pandas as pd
from loguru import logger
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.utils.dataframe import dataframe_to_rows

from conf.config import DATA_DIR, VALID_COLUMNS

def load_data(path=DATA_DIR / 'mgx_ensembling_fix_exexutability.xlsx') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载并预处理数据

    返回:
        Tuple[pd.DataFrame, pd.DataFrame]: 处理后的数据和版本数据
    """
    # 加载主数据
    df = pd.read_excel(path)
    df = df[VALID_COLUMNS]
    
    # 填充缺失的可执行性值
    # df['executability'] = df['executability'].fillna(0.2)
    df['executability'] = df['executability'].fillna(0.0)
    
    # 将llm_score转换为float类型
    df['llm_score'] = df['llm_score'].astype(float)
    # 归一化数据
    columns_to_normalize = ['cost($)', 'project_files_count', 'code_files_count', 'code_lines']
    df = normalize_data(df, columns_to_normalize, 'case_name')
    
    return df

def normalize_data(df: pd.DataFrame, columns_to_normalize: List[str], group_by: str) -> pd.DataFrame:
    """按组归一化数据"""
    for case_name, group in df.groupby(group_by):
        for col in columns_to_normalize:
            if col not in df.columns:
                continue
                
            group_data = group[col].fillna(0)
            non_zero_values = group_data[group_data > 0]
            
            if not len(non_zero_values):
                df.loc[group.index, f'{col}_normalized'] = 0
                continue
                
            non_zero_min = non_zero_values.min()
            max_val = group_data.max()
            
            df.loc[group.index, f'{col}_normalized'] = (
                group_data.apply(lambda x: 0 if x == 0 else (x - non_zero_min) / (max_val - non_zero_min))
                if max_val > non_zero_min else 0
            )
            
            logger.info(f"Case {case_name} - 已归一化 {col}")
    
    return df


def create_valid_sheet_name(name: str, existing_names: List[str]) -> str:
    """创建有效的Excel工作表名称"""
    sheet_name = str(name)[:30]
    sheet_name = sheet_name.translate(str.maketrans({c: '_' for c in '/?*[]:\\'}))
    
    base_name = sheet_name
    return next(
        f"{base_name}_{i}" if i else base_name
        for i in range(len(existing_names) + 1)
        if f"{base_name}_{i}" not in existing_names
    )


def save_dataframe_to_excel(df: pd.DataFrame, worksheet: Worksheet) -> None:
    """
    将DataFrame保存到Excel工作表
    
    参数:
        df: 要保存的DataFrame
        worksheet: 目标工作表
    """
    for r in dataframe_to_rows(df, index=False, header=True):
        worksheet.append(r)


def get_reward_methods(calculator_class) -> List[str]:
    """获取计算器类中的所有奖励计算方法"""
    return [m for m in dir(calculator_class) if m.startswith('calculate_reward')]


def get_reward_numbers(calculator_class) -> List[str]:
    """获取计算器类中的所有奖励函数编号"""
    return [m.replace('calculate_reward', '') for m in get_reward_methods(calculator_class)] 