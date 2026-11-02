# 实验四文件索引

四项实验均已在本机 CPU 上实际训练、评估并保留 Notebook 输出。结果图片为 PNG，指标及预测按需保存为 CSV 或 JSON；本文件用于定位产物，不是 Word 报告正文。

| 实验 | Notebook | 实际测试结果 | 指标文件 |
| --- | --- | --- | --- |
| A 深圳天气预测 | [exp4A.ipynb](exp4A.ipynb) | 7 天 LSTM：温度 MAE 1.1279℃、RMSE 1.5151℃；天气 Accuracy 0.6298、宏 F1 0.6388 | [metrics.csv](results/exp4A/metrics.csv) |
| B 中文情感三分类 | [exp4B.ipynb](exp4B.ipynb) | 多层 LSTM：Accuracy 0.8766、宏 F1 0.8630；BiLSTM：0.8750、0.8609 | [metrics.csv](results/exp4B/metrics.csv) |
| C 中文图像描述 | [exp4C.ipynb](exp4C.ipynb) | 字符级 BLEU4 0.0849、ROUGE L 0.3016；错配图像对照分别为 0.0262、0.1802 | [metrics.csv](results/exp4C/metrics.csv) |
| D 双尺度 YOLO 检测 | [exp4D.ipynb](exp4D.ipynb) | Precision 0.4796、Recall 0.3745、mAP@0.5 0.2208；P/R 为微平均 | [metrics.csv](results/exp4D/metrics.csv) |

## 报告配图与结果

| 实验 | 关键图片 | 预测与详细结果 |
| --- | --- | --- |
| A | [数据分布](results/exp4A/weather_data.png)、[训练曲线](results/exp4A/training_curves.png)、[气温预测](results/exp4A/temperature_predictions.png)、[天气混淆矩阵](results/exp4A/weather_confusion.png) | [测试预测](results/exp4A/test_predictions.csv)、[训练记录](results/exp4A/training_history.csv)、[数据来源](results/exp4A/data_source.json) |
| B | [训练曲线](results/exp4B/training_curves.png)、[混淆矩阵](results/exp4B/confusion_matrices.png)、[模型代码图](results/exp4B/model_code.png) | [测试预测](results/exp4B/test_predictions.csv)、[三类映射](results/exp4B/label_mapping.csv)、[数据划分](results/exp4B/data_split.csv) |
| C | [训练曲线](results/exp4C/training_curves.png)、[六张描述样例](results/exp4C/caption_examples.png)、[模型代码图](results/exp4C/model_code.png) | [全部测试生成文本](results/exp4C/test_predictions.csv)、[图像划分](results/exp4C/image_split.csv) |
| D | [训练曲线](results/exp4D/training_curves.png)、[六组检测图](results/exp4D/detection_examples.png)、[PR 曲线](results/exp4D/precision_recall.png)、[遮挡对照](results/exp4D/occlusion_example.png)、[模型代码图](results/exp4D/model_code.png) | [各类指标](results/exp4D/class_metrics.csv)、[尺度召回](results/exp4D/scale_recall.csv)、[遮挡压力测试](results/exp4D/occlusion_stress.csv)、[预测框](results/exp4D/test_predictions.json)、[图像划分](results/exp4D/image_split.csv) |

C、D 的六组样例均另存为各自结果目录内的 `example_1.png` 至 `example_6.png`，便于单独插入报告。代码图由实际 Notebook 的模型代码排版生成，可替代手动截取的代码配图。

## 数据与运行方式

从仓库根目录或 `exp4/` 目录启动 Notebook，按顺序运行所有单元格。依赖在根目录 `requirements.txt` 中；本次解释器为 `D:\Anaconda\python.exe`，Python 3.14.6、PyTorch 2.14.1+cpu。数据在 `exp4/data/`，权重和确定性特征缓存位于 `exp4/.cache/`，均已忽略。首次缺少数据时会下载，后续复用缓存；模型训练单元格仍会重新训练。

A 使用 [Open-Meteo 历史接口](https://open-meteo.com/en/docs/historical-weather-api) 的深圳 ERA5 网格再分析数据。原指定 ip.cn 连接失败后，用户已确认使用可获取的历史数据。日期为 2024-01-01 至 2026-10-16，2024 年训练、2025 年验证、2026 年测试，目标日期分别为 336、365、289 天。2026 年不是完整年度，数据也不是气象站逐日实测；全部模型按已知过去观测进行下一天预测。

B 使用 [指定中文情感数据集](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)，4159 条原始记录，繁简转换及去重后 4154 条，训练/验证/测试为 2907/623/624。八类到三类的映射是实验约定，关切、惊奇的极性存在语境歧义。两种模型均有过拟合迹象，使用验证宏 F1 选择权重。

C 使用 [cross-lingual-cap 所列 COCO-CN](https://github.com/li-xirong/cross-lingual-cap)，保留官方划分，在各集合分别随机取 3000/300/300 张图。冻结 ImageNet 预训练 ResNet18，训练字符级 LSTM。生成样例保留了颜色、动作错误和重复表达；字符级 BLEU、ROUGE 不与论文分词级成绩直接比较。

D 使用 [Hard Hat Detection 公开镜像](https://huggingface.co/datasets/Voxel51/hard-hat-detection) 的 1600 张图，按近重复图像分组后训练/验证/测试为 1119/240/241。教学用轻量 YOLO 全部权重随机初始化，完成 45 轮训练，最低验证损失在第 18 轮。person 类在所选操作阈值下召回为 0，AP50 为 0.0084，结果不满足实际部署要求。原始 occluded 属性全部为 0，因此遮挡部分采用明确标注的合成灰块压力测试，召回由 0.3745 降至 0.1380；大目标测试支持数仅 4，不能得出稳定的尺度优劣结论。

四项结果已核对顺序执行、错误输出、数据划分和主要指标与保存预测的一致性。图片已检查中文显示和排版。Word 模板保持原样；未提交 Git 或推送。
