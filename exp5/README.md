# 实验五文件索引

前三部分已完成训练、评价及注意力可视化；第四部分仅完成人脸检测和条件压力测试，**未完成身份识别，不能作为四部分全部完成的提交版本**。身份识别实现及匿名编号识别实现均被自动审批拒绝，工具未提供具体原因。未修改报告正文，未提交Git。

| 部分 | Notebook | 状态与结果 |
| --- | --- | --- |
| A 自注意力对比 | [exp5A.ipynb](exp5A.ipynb) | 第1/6/12层、12层深度加权图，逐层未来注意力比例与熵；GPT因果掩码检查通过 |
| B BERT情感分类 | [exp5B.ipynb](exp5B.ipynb) | Accuracy 0.8990，宏Precision 0.8873，宏Recall 0.8917，宏F1 0.8890 |
| C GPT-1条件生成 | [exp5C.ipynb](exp5C.ipynb) | Accuracy 0.6266，宏Precision 0.6371，宏Recall 0.5922，宏F1 0.6067；624条中2条非法生成计为错误 |
| D 人脸检测（部分完成） | [exp5D.ipynb](exp5D.ipynb) | LFW随机300张，原图检出比例1.000；其余条件见CSV。不是身份识别Accuracy或检测Precision/Recall |

## 配图与数值结果

| 部分 | 关键图片 | 数值结果 |
| --- | --- | --- |
| A | [BERT注意力](results/exp5A/bert_attention.png)、[GPT注意力](results/exp5A/gpt_attention.png) | [注意力统计](results/exp5A/attention_statistics.csv) |
| B | [损失曲线](results/exp5B/training_curves.png)、[性能](results/exp5B/performance.png)、[训练前后注意力](results/exp5B/attention_before_after.png)、[实际代码排版图](results/exp5B/model_code.png) | [指标](results/exp5B/metrics.csv)、[预测](results/exp5B/test_predictions.csv)、[训练记录](results/exp5B/training_history.csv) |
| C | [损失曲线](results/exp5C/training_curves.png)、[性能](results/exp5C/performance.png)、[训练前后注意力](results/exp5C/attention_before_after.png)、[横向比较](results/exp5C/model_comparison.png)、[实际代码排版图](results/exp5C/model_code.png) | [指标](results/exp5C/metrics.csv)、[预测与生成文本](results/exp5C/test_predictions.csv)、[训练记录](results/exp5C/training_history.csv)、[横向比较](results/exp5C/model_comparison.csv) |
| D | [检测与扰动对照](results/exp5D/detection_conditions.png)、[检出比例](results/exp5D/detection_coverage.png) | [检测统计](results/exp5D/detection_metrics.csv)、[逐图检测](results/exp5D/detection_details.csv) |

代码配图由Notebook实际代码排版生成，不是手动截屏。B/C保存三分类报告、相同的数据划分和标签映射；注意力原始数组按需保存为NPZ。[模型文件SHA256清单](results/model_sources.json)记录此次权重和配置。

## 数据、方法和限制

情感数据使用指定的 [Chinese Multi-Emotion Dialogue Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)。原始4159条，繁简转换、去重和冲突排除后4154条；训练/验证/测试2907/623/624。开心映射正面，悲伤/愤怒/厌恶映射负面，其余映射中性；这是教学映射，惊奇、关切等存在语境歧义。固定随机种子42，相同数据划分，测试集不参与选模。

[BERT-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)与[GPT-1](https://huggingface.co/openai-community/openai-gpt)都保留完整12层结构。为控制8GB显存和训练时间，BERT训练最后4层、池化层和分类头，GPT训练词嵌入及最后4层。BERT训练5轮（3e-5、batch16），GPT训练6轮（Transformer 3e-5、嵌入2e-4、batch12）；均按验证宏F1选择权重，最优分别第5/6轮。使用AdamW和GPU BF16。训练曲线保留实际过拟合现象。

GPT-1原始英语词表对选定中文Sentence的UNK比例为100%。因此从训练集提取中文字符扩展词表，新增嵌入随机初始化，逐字编码；不替换GPT-1主体。输入BOS+中文文本+sentiment，监督输出positive/negative/neutral专用标签token和EOS，输入位置不计损失；完整词表贪心生成，非法标签或未生成EOS计为错误。BERT与GPT在预训练语言、分词和训练预算上不同，横向指标不能解释为纯结构优劣。

同一句Sentence为“今天终于通过了考试，我真的很开心！”。单层先平均12个头，多层使用第l层权重l/78；固定深度加权不是可训练权重，也不是attention rollout。注意力不是因果解释，特殊token占据较大权重的真实现象予以保留；跨模型分词位置不同，不逐格直接相减。

D的原LFW figshare下载入口403，改从同一数据集的[bitmind/lfw公开镜像](https://huggingface.co/datasets/bitmind/lfw)读取13233张250×250图片，仅抽取固定随机300张用于检测。使用[OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)官方预训练检测模型，score阈值0.8、NMS阈值0.3。没有身份分类，也无人工框标注，检出比例仅代表图中检出至少一张人脸。亮度×0.45/×1.5、平面旋转20度、固定灰块都是合成扰动；平面旋转不代表三维侧脸，不能泛化为真实光照/遮挡识别性能。图片本身已以脸为中心。

## 运行

本次环境：Windows，Python 3.14.6，PyTorch 2.14.1+cu130，RTX 5060 8GB。原仓库.venv失效，原系统PyTorch为CPU版，因此独立GPU环境位于仓库根目录`.cache/exp5-env`，未替换系统环境。可在Jupyter中选择该环境；缺少环境时按根目录requirements.txt安装依赖，CUDA版PyTorch需要匹配显卡单独安装。CPU也可运行，但完整训练耗时较长。

从仓库根目录或exp5目录启动，按A、B、C、D顺序执行各Notebook。C最后的横向比较依赖B的结果。数据下载到data/，权重和缓存位于.cache/，均已忽略；缺少模型文件时自动从指定仓库下载。首次运行需要网络。本地训练单元格重新训练，已有结果不会用来冒充新运行。
