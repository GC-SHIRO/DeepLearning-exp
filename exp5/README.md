# 实验五文件索引

前三部分完成自注意力对比、BERT情感分类与GPT-1条件生成，使用SST-5数据并保留实际运行输出。第四部分仅完成人脸检测与条件压力测试，尚未完成身份识别及其评价指标。本文件用于定位实验产物，不是报告正文。

| 部分 | Notebook | 结果 |
| --- | --- | --- |
| A 自注意力对比 | [exp5A.ipynb](exp5A.ipynb) | 第1/6/12层及12层加权，原生英语分词，GPT因果掩码核验 |
| B BERT分类 | [exp5B.ipynb](exp5B.ipynb) | Accuracy 0.7421，宏Precision 0.6542，宏Recall 0.6446，宏F1 0.6410；最优第4轮 |
| C GPT-1条件生成 | [exp5C.ipynb](exp5C.ipynb) | Accuracy 0.7389，宏Precision 0.6736，宏Recall 0.6735，宏F1 0.6733；最优第3轮 |

## 配图与结果文件

| 部分 | 图片 | 数值结果 |
| --- | --- | --- |
| A | [BERT注意力](results/exp5A/bert_attention.png)、[GPT注意力](results/exp5A/gpt_attention.png) | [逐层统计](results/exp5A/attention_statistics.csv) |
| B | [损失曲线](results/exp5B/training_curves.png)、[测试混淆矩阵](results/exp5B/performance.png)、[训练前后注意力](results/exp5B/attention_before_after.png)、[训练后分层图](results/exp5B/attention_after.png)、[实际代码排版图](results/exp5B/model_code.png) | [指标](results/exp5B/metrics.csv)、[训练记录](results/exp5B/training_history.csv)、[全部测试预测](results/exp5B/test_predictions.csv)、[各类指标](results/exp5B/class_metrics.csv) |
| C | [损失曲线](results/exp5C/training_curves.png)、[测试混淆矩阵](results/exp5C/performance.png)、[训练前后注意力](results/exp5C/attention_before_after.png)、[训练后分层图](results/exp5C/attention_after.png)、[横向对比](results/exp5C/model_comparison.png)、[实际代码排版图](results/exp5C/model_code.png) | [指标](results/exp5C/metrics.csv)、[训练记录](results/exp5C/training_history.csv)、[全部生成结果](results/exp5C/test_predictions.csv)、[对比指标](results/exp5C/model_comparison.csv) |

代码图片是实际Notebook代码的排版图，长行自动折行；不是手动截图。两模型均保存Sentence预测、注意力数组和截断统计。

## 数据和模型

使用[SetFit/SST-5](https://huggingface.co/datasets/SetFit/sst5)的完整句子级英文影评数据，共11855条。标签0/1映射negative，2映射neutral，3/4映射positive，内部编号依次为negative=1、neutral=2、positive=0。清理空白并小写，排除冲突、去除重复；保留官方划分，跨集合重复优先留在测试集、再验证集。最终11841条：训练8531、验证1100、测试2210；并非抽取小子集，集合间无重复文本。数据属于电影评论，结果仅代表此类文本的情感分类表现。

使用[google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)和[openai-community/openai-gpt](https://huggingface.co/openai-community/openai-gpt)。相同Sentence为：`I loved this wonderful movie and felt very happy.`。GPT原生英语词表对该句UNK比例为0%，保留原始语言词表，只新增padding和EOS控制标记。标签positive/negative/neutral是原词表中的单token。

两模型都保留完整12层，固定种子42，batch16、AdamW学习率3e-5、weight_decay=0.01，训练4轮，按验证宏F1选择权重。BERT训练最后4层、池化层及分类头，最大输入96token；GPT训练词嵌入及最后4层，文本最多80个BPE token，另外保留`Review:`和`Sentiment:`提示与输出空间。截断统计各见`truncation.json`，原生分词不同，不能假设实际文本覆盖完全相同。GPT只对标签和EOS计算损失，完整词表贪心生成，非法输出计错。验证和测试的批量生成分别取每条序列的实际末位置，不把padding当作生成位置。

注意力先对12个头求均值，跨层使用第l层权重l/78。BERT分析纯Sentence，GPT分析包含同一句Sentence的Review/Sentiment提示。前后分析保持各模型自身输入完全一致；A与B/C的训练前数组核对一致。不同模型分词和提示不同，不能逐格相减；层间固定加权不是attention rollout，也不是因果解释。

## 实际限制

BERT验证损失后期上升，中性类别召回较低；GPT第4轮验证损失及宏F1变差，因此选择第3轮。GPT的宏F1在本设置下较高，BERT的Accuracy略高，结果不支持笼统认定某一种结构全面更好。两者训练参数量、预训练目标、词表和输入形式不同，不能解释为严格控制其他条件的结构消融。

## 运行和核验

从仓库根目录或exp5目录启动，依次运行A、B、C，C末尾对比需要B的指标文件。依赖沿用根目录requirements.txt。实际环境为Windows、Python3.14.6、PyTorch2.14.1+cu130、RTX5060 8GB，解释器在仓库`.cache/exp5-env/Scripts/python.exe`。CPU可运行但完整训练较慢。

下载数据位于`data/sst5/`，预训练模型及本次权重位于`.cache/`，均已忽略；训练权重保存为`bert_finetuned.pt`和`gpt_finetuned.pt`。Notebook缺少数据或原始模型文件时自动下载；训练单元格会重新训练。模型并非直接加载已经完成情感微调的第三方权重。

[核验记录](results/verification.json)包括全部测试预测重算指标、相同数据划分、跨集合无重复文本、选模轮次、注意力数组及Notebook执行状态。[数据和模型SHA256清单](results/sources.json)记录本次源文件。已实际查看结果图片。未修改Word报告、未提交Git。

## 人脸检测部分

[exp5D.ipynb](exp5D.ipynb)仅完成检测和合成条件测试，未完成身份识别。配图见[检测条件对照](results/exp5D/detection_conditions.png)，数值见[检测统计](results/exp5D/detection_metrics.csv)。

D的原LFW figshare下载入口403，改从同一数据集的[bitmind/lfw公开镜像](https://huggingface.co/datasets/bitmind/lfw)读取13233张250×250图片，仅抽取固定随机300张用于检测。使用[OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)官方预训练检测模型，score阈值0.8、NMS阈值0.3。没有身份分类，也无人工框标注，检出比例仅代表图中检出至少一张人脸。亮度×0.45/×1.5、平面旋转20度、固定灰块都是合成扰动；平面旋转不代表三维侧脸，不能泛化为真实光照/遮挡识别性能。图片本身已以脸为中心。
