# 实验六 · 生成对抗网络与 Stable Diffusion LoRA

## 任务与验收

| 部分 | 模板要求 | 本次设计与验收依据 |
| --- | --- | --- |
| A 风格迁移 | 训练从自然图像到艺术风格图像的 GAN，对比不同损失权重，结合损失和生成图随训练的变化分析稳定性、风格与内容 | [exp6A.ipynb](exp6A.ipynb) 在相同的 256 张训练照片上分别训练 λ=1 和 50 的条件 GAN。保留第 1、10、40 轮同一张图的输出，在 40 张独立测试照片上计算 MAE、SSIM 和色号预测准确率。 |
| B 个性化生成 | 说明文本编码器、U-Net、VAE 和 LoRA 的原理与参数，记录数据筛选、预处理、Caption 构建过程，用多个 Prompt 对比微调前后的效果 | [exp6B.ipynb](exp6B.ipynb) 用 8 种主体各 6 张图构建六色版画风格集。冻结 SD 1.5 原权重，微调 U-Net 注意力层的秩 4 适配器。三组提示分别描绘窗台的猫、过桥的火车和海边灯塔，每对生成图共用种子与采样参数。 |

模板给出了需要比较和解释的现象，具体数据量、损失权重、训练轮数与测试提示词由本实验选择。Word 中的思考题评分栏没有附具体题目。

## 数据与运行

[图像清单](data_index.csv) 保存每张照片的来源、SHA-256、实验四 COCO-CN 划分、中文描述和 LoRA 英文 Caption。训练和测试按图像 ID、文件哈希分开。COCO 原始文件名中的 train2014、val2014 表示照片来源，本实验划分以清单 split 列为准。[COCO](https://cocodataset.org/) 提供照片，[COCO-CN](https://huggingface.co/datasets/AIMClab/COCO-CN) 提供中文描述。[风格处理代码](style_data.py) 对照片中心裁剪，经过中值滤波、亮度分段后映射到六种印刷色。这种规则提供了空间对齐的监督目标，适用于本次限定的版画风格。

在安装了仓库依赖的 Python 内核中，分别从头运行两份 Notebook。可以从仓库根目录或实验六目录启动。脚本优先复用实验四缓存，缺图时按清单下载到 `exp6/data/`。B 首次运行需要取得固定版本的 [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)。基础模型和训练生成的适配器均缓存在仓库 `.cache/exp6/`。A 与 B 可分别运行。

## A 的结果

| 测试尺寸 | L1 权重 | 测试 MAE ↓ | 测试 SSIM ↑ | 像素色号正确率 ↑ |
| ---: | ---: | ---: | ---: | ---: |
| 96×96 | 1 | 0.0216 | 0.7825 | 88.32% |
| 96×96 | 50 | 0.0202 | 0.7919 | 89.15% |
| 192×192 | 1 | 0.0182 | 0.8085 | 90.05% |
| 192×192 | 50 | 0.0170 | 0.8176 | 90.74% |

生成器预测每个位置的六种颜色得分，展示时取最高分对应的色号。固定测试样本中，路面形成连续色块，红绿灯轮廓清晰。人物腿部和树叶等边缘仍有色号偏差。原版在 96×96、λ=50 下的 SSIM 为 0.6880，新版在相同测试条件下达到 0.7919。模型结构、色号监督、训练输入和轮数均有调整，这次对照衡量整体方案的变化。[原版对照图](results/exp6A/baseline_rgb96/comparison.png) · [原版指标](results/exp6A/baseline_rgb96/metrics.csv)

[测试对照](results/exp6A/comparison.png) · [训练演化](results/exp6A/evolution.png) · [训练曲线](results/exp6A/loss.png) · [汇总指标](results/exp6A/metrics.csv) · [逐图指标](results/exp6A/per_image.csv)

## B 的结果

LoRA 训练了 144 步，每张图参与训练 3 次，更新 797,184 个参数，约占适配后 U-Net 参数的 0.093%。在同一随机种子下，灯塔图由渐变海浪变成蓝绿和浅橙色块，最近色距离从 0.2760 降到 0.1676。猫图也出现较大色块，脸部细节有所丢失。过桥火车一组的距离仅从 0.2469 降到 0.2430，前后都能看到桥梁结构，列车主体都难以辨认。风格接近程度和提示词符合程度需分别观察。这三组单种子结果只支持对展示样例的分析。

[训练集示例](results/exp6B/training_examples.png) · [微调前后对照](results/exp6B/prompt_comparison.png) · [训练曲线](results/exp6B/loss.png) · [逐提示词指标](results/exp6B/prompt_metrics.csv) · [训练参数](results/exp6B/settings.json)
