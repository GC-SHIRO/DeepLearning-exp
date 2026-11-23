# 实验七　文本情感识别

[实验 Notebook](exp7A.ipynb) 包含全部训练与评估，可从仓库根目录或实验目录顺序运行。原始数据按固定版本缓存在 `exp7/data/`，模型权重保存在仓库 `.cache/exp7/`。随机种子为 707，分类阈值为正面概率大于 0.5。每个配置训练 4 轮，通过验证集宏 F1 选择权重。

| 实验内容 | 配图 | 可复查的数据 |
| --- | --- | --- |
| RNN、BiLSTM-Attention、单层 Transformer、BERT-Tiny 域内分类 | [八组混淆矩阵](results/exp7A/confusion.png) | [指标](results/exp7A/metrics.csv) · [逐条预测](results/exp7A/predictions.csv) · [错误原文](results/exp7A/error_examples.csv) |
| IMDb 与 SST-2 双向迁移 | [同域及跨域宏 F1](results/exp7A/transfer.png) | [样本与截断比例](results/exp7A/data_summary.csv) · [划分清单](results/exp7A/split_manifest.csv) |
| 隐藏宽度、注意力头数、嵌入初始化和位置编码 | [四组受控消融](results/exp7A/ablation.png) | [消融结果](results/exp7A/ablation.csv) · [训练记录](results/exp7A/training.csv) |
| 近义词、插入从句和显式否定 | [成对正确率](results/exp7A/robustness.png) · [注意力可视化](results/exp7A/attention.png) | [扰动逐条预测](results/exp7A/probe_predictions.csv) · [扰动汇总](results/exp7A/robustness.csv) · [注意力权重](results/exp7A/attention.csv) |

## 读数与限制

主实验中 BERT-Tiny 的 IMDb 准确率为 77.70%，宏 F1 为 0.7759，SST-2 准确率为 79.36%，宏 F1 为 0.7931。IMDb 负评中有 292 条被判为正面，正评中有 154 条被判为负面。SST-2 错例中，有一条评价先描述沉浸在视觉效果中，末尾表达难以理解电影主旨，BERT 将该负评判为正面。

IMDb 训练的 BiLSTM-Attention 迁移到 SST-2 的宏 F1 为 0.7291。反向迁移时为 0.6970。两套训练集各分层抽取 4,800 条，IMDb 评估集使用官方测试集中的 2,000 条。SST-2 官方验证集等分为调参和最终评估两组，各 436 条。两套语料都涉及电影，评估范围覆盖长影评与短评之间的迁移。输入上限为 96 个 WordPiece，IMDb 测试长评有 92.8% 被截断。

在 SST-2 消融中，随机初始化词嵌入后，自建 Transformer 宏 F1 从 0.7676 降至 0.6650。两头的结果比四头高 0.0029，仅进行了单种子比较。SST-2 BERT 在 16 组同义替换中全部成对正确，在 16 组显式否定中成对正确 5 组。人工短句只覆盖所列词汇和句式。注意力图用于查看各位置的权重分布，对词语是否改变决策的判断还需要结合成对预测。
