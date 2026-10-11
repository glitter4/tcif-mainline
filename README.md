# TCIF

CMU-MOSEI、CMU-MOSI 和 CH-SIMS 的 Temporal Context Innovation Filter 训练与推理实现。

模型由文本、视觉和音频编码器，模态专属／共享专家，两条任务分支，以及分别作用于回归和分类特征的 TCIF 组成。训练包含中心样本回归与分类、上下文辅助监督和转移门控监督。文本不访问共享专家；相邻片段正常参与反向传播。

## 配置与对应结果

| 配置 | 文本编码器 | 可训练文本范围 | seed | 权重 epoch | Acc7 / Acc5 (%) | MAE | Acc2 nonzero (%) | weighted F1 nonzero (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `configs/mosei.json` | RoBERTa-base | 最后 2 个 encoder blocks | 40 | 4 | 56.664520 | 0.492930 | 87.671987 | 87.681209 |
| `configs/mosi.json` | BERT-base-uncased | 全部文本参数，包括 embeddings | 123 | 57 | 48.542274 | 0.701974 | 84.603659 | 84.518509 |
| `configs/chsims.json` | RoBERTa-base | 最后 1 个 encoder block | 41 | 28 | 49.452954 | 0.388044 | 83.505155 | 83.676047 |

表中数值来自对应的既有权重。MOSEI/MOSI 使用七分类概率期望，CH-SIMS 使用五分类最大概率类别的分值；推理均固定为 `0.2 × regression + 0.8 × classification`，分类温度为 1。Acc7 使用最近偶数舍入；二分类指标排除真实标签为零的样本。

`epochs` 是原学习率计划的完整长度，`save_epoch` 是本配置训练至并输出的权重轮次。MOSI 的 cosine 计划长度仍为 150，CH-SIMS 为 50。缩短运行至指定轮次时，学习率计划长度保持原值。每个配置保存一个明确轮次的权重文件。

## 环境和输入

使用 Python 3.10，安装与计算设备相符的 PyTorch 2.7.1 / torchvision 0.22.1 / torchaudio 2.7.1，然后安装依赖：

```bash
python -m pip install -r requirements.txt
```

准备本地 Hugging Face 模型目录：上述对应文本编码器、`vit-base-patch16-224-in21k` 和 `hubert-base-ls960`。模型权重通过显式路径加载。

数据目录采用原处理格式：

```text
dataset/
  label.npz
  transcription-engchi-polish.csv
cache/
  train/<shard>/
  val/<shard>/
  test/<shard>/
```

`label.npz` 包含 `train_corpus`、`val_corpus`、`test_corpus`，每个字典以样本 ID 为键，`val` 为连续标签。CSV 保留原样本 ID 与英文转写列；CH-SIMS 配置也使用原实验的英文转写输入。上下文仅从同一 split 的同一视频序列中建立，中心样本不作为有效邻居。

每个完成的缓存 shard 包含 `ids.json`、`done.json`、`vision.npy`、`audio.npy`、`audio_mask.npy`、`text.npy`、`text_mask.npy`，数组首维与 `ids.json` 顺序一致。视觉输入保留每段 4 帧及比例 0.2 的相邻帧上下文。文本始终由当前可训练编码器在线计算，缓存文本不进入模型。需要使用与原数据处理、模型和采样设置一致的缓存。

数据、特征缓存、预训练模型和训练所得权重由运行环境提供。MOSI 使用 MOSI 自身的训练数据初始化和训练，数据划分为 1284 / 229 / 686；CH-SIMS 为 1368 / 456 / 457。

## 训练

以 MOSEI 为例，将参数替换为实际路径：

```bash
python train.py \
  --config configs/mosei.json \
  --data-root /path/to/mosei \
  --cache-root /path/to/mosei-cache \
  --text-model /path/to/roberta-base \
  --vision-model /path/to/vit-base-patch16-224-in21k \
  --audio-model /path/to/hubert-base-ls960 \
  --output outputs/mosei.pth
```

MOSI 和 CH-SIMS 使用各自 JSON 与对应的数据、缓存和文本模型目录。训练沿用配置中的微 batch、梯度累积、优化器分组和文本解冻范围。MOSI 的邻居 dropout 为 0.2，仅在训练时作用于上下文有效性掩码。

CH-SIMS 的 `temporal_projection` 保留原权重中的投影层和初始化顺序；训练目标由上述四项损失组成。

## 推理

```bash
python infer.py \
  --config configs/mosei.json \
  --data-root /path/to/mosei \
  --cache-root /path/to/mosei-cache \
  --text-model /path/to/roberta-base \
  --vision-model /path/to/vit-base-patch16-224-in21k \
  --audio-model /path/to/hubert-base-ls960 \
  --checkpoint /path/to/mosei.pth \
  --split test \
  --output outputs/mosei-predictions.csv
```

推理严格检查权重键名和张量形状，输出逐样本 `id,label,prediction` CSV，并打印指标。两个入口默认使用 CUDA；CPU 运行可加 `--device cpu`。输出路径已存在时会报错，避免覆盖。

## 文件

| 文件 | 用途 |
|---|---|
| `train.py` | 训练循环、学习率计划和固定轮次权重保存 |
| `infer.py` | 指定权重推理与指标计算 |
| `tcif/model.py` | 三模态编码、专家层、任务读出 |
| `tcif/filter.py` | 上下文先验、转移门控与精度加权更新 |
| `tcif/data.py` | 三数据集、特征缓存、相邻片段输入 |
| `tcif/readout.py` | 标签离散化与固定推理读出 |
| `tcif/training.py` | 训练损失和优化器参数分组 |
| `tcif/runtime.py` | 配置、随机种子、模型和数据构建 |

随机种子和初始化顺序按原链路保留；不同设备、库版本及非确定性算子可能影响重新训练的数值。
