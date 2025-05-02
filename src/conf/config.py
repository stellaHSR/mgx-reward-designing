# -*- coding: utf-8 -*-
# @Author  : 
# @Desc    :
import os
from pathlib import Path

DATA_DIR = Path(os.path.dirname(__file__)).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "output"
# 配置常量
SAMPLE_SIZES = [3, 5, 8, 10, 12, 16, 20]
VALID_COLUMNS = [
    'case_name', 'scenario', 'llm_score', 'human_score',
    'executability', 'quality', 'model', 'total_time',
    'input_tokens', 'completion_tokens', 'input_cost',
    'completion_cost', 'interactions_count', 'cost($)',
    'prod_url', 'project_files_count', 'code_files_count', 'code_lines'
]

