# mgx-reward-designing

一个用于分析和评估AI奖励函数效果的Python工具包。支持多种奖励计算方法的性能评估,并提供详细的分析报告和可视化结果。

## 项目结构

```
benchmark_data/mgx-reward-designing/
├── src/
│   ├── bon.py              # 主程序文件
│   ├── ensemble_processor.py  # 集成处理器
│   ├── reward_func.py      # 奖励函数定义
│   └── utils/
│       ├── utils.py        # 通用工具函数
│       └── visualization_utils.py  # 可视化工具
├── conf/
│   └── config.py          # 配置文件
└── output/                # 输出目录
```

## 主要功能

- 多种奖励函数的计算与对比分析
- 支持不同样本大小的自适应分析
- 生成Excel分析报告和可视化图表
- 支持集成奖励计算
- 与人类评分的对比分析
- 自动错误检测和异常数据高亮

## 快速开始

### 环境要求
- Python 3.6+
- 依赖包: pandas, numpy, openpyxl, loguru, matplotlib

### 安装依赖
```bash
pip install pandas numpy openpyxl loguru matplotlib
```

### 运行分析
1. 在 `conf/config.py` 中配置参数
2. 运行分析程序:
```bash
python src/bon.py
```

## 输出文件说明

### 1. case_results.xlsx
- 按样本大小分sheet展示分析结果
- 包含各奖励函数得分、集成得分和人类评分
- 自动高亮异常数据

### 2. reward_trends.png
- 展示不同样本大小下的奖励方法表现趋势
- 包含平均得分、标准差和人类评分基准线

### 3. reward_analysis.log
- 记录程序运行日志和警告信息

## 配置说明

在 `conf/config.py` 中可配置:
- OUTPUT_DIR: 输出路径
- SAMPLE_SIZES: 分析样本大小列表
- 其他参数...

## 注意事项

- 确保输入数据完整性和格式正确
- 样本大小不应超过原始数据量
- 使用固定随机种子(49)确保结果可复现

## 开发指南

如需扩展功能:
1. 新增奖励函数: 在 `reward_func.py` 中添加
2. 添加分析维度: 扩展 `RewardAnalyzer` 类
3. 自定义可视化: 修改 `visualization_utils.py`

## 许可证

MIT License

