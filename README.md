# Text Classification Project (工程化版本)

> 包含：数据集构建/清洗 → TF-IDF baseline → BiLSTM 分类 → BiLSTM+Attention → 消融实验 → 错误分析  
> 目标：**任何人 clone 后，在一条命令下可以复现结果**（可控随机种子、可追溯配置、统一输出目录）。

---

## 1. 目录结构（你要能背出来 + 能解释为什么这样分）

```
nlp_textclf_project/
├─ configs/                      # 所有实验都由 YAML 配置驱动（可复现的核心）
├─ data/
│  ├─ raw/                       # 原始数据放这里（不进 git）
│  ├─ processed/                 # 清洗后的数据（不进 git）
│  └─ splits/                    # train/val/test 划分（不进 git）
├─ notebooks/
│  └─ original/                  # 你原来的 ipynb（只做存档/展示）
├─ scripts/                      # 入口脚本（只做“组装”与参数解析）
├─ src/nlp_textclf/              # 可复用的核心代码（数据/模型/训练/分析）
│  ├─ data/
│  ├─ models/
│  ├─ training/
│  ├─ analysis/
│  └─ utils/
├─ outputs/                      # 每次运行会生成一个输出目录（不进 git）
├─ tests/                        # 最小单测（保证关键函数不炸）
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

**核心原则：**
- `scripts/` 只做“拼装 + 调参”，不要写核心逻辑；
- 核心逻辑放 `src/`，保证可复用、可测试；
- 一切可变内容（超参、路径、seed）放 `configs/`，保证可追溯；
- `outputs/` 里每个 run 都保存：`config.yaml + vocab/model + metrics`，保证复盘。

---

## 2. 环境安装

```bash
# 1) 建议新建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Windows 用 .venv\Scripts\activate

# 2) 安装依赖
make setup
```

---

## 3. 数据（两种方式）

### 方式 A：直接用 IMDb（推荐先跑通工程）
不需要你准备数据，会自动下载（HuggingFace datasets）。

- BiLSTM / Attention / Ablation / Error Analysis 默认使用 IMDb 配置。

### 方式 B：用你自己的 CSV 数据（你 notebook 的场景）
把数据放到：`data/raw/raw_data.csv`  
格式要求：

| 列名 | 含义 |
|---|---|
| text | 文本 |
| label | int 标签（0..K-1） |
| source_file（可选） | 如果样本来自不同文件，可用它避免数据泄漏 |

然后执行：

```bash
make build-data
```

会生成：
- `data/processed/cleaned.csv`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

---

## 4. 训练与评测

### 4.1 TF-IDF + LogisticRegression（baseline）
```bash
make train-tfidf
```

输出在：
- `outputs/tfidf_lr/metrics.json`
- `outputs/tfidf_lr/val_report.txt`
- `outputs/tfidf_lr/test_report.txt`
- `outputs/tfidf_lr/tfidf_vectorizer.joblib`
- `outputs/tfidf_lr/logreg.joblib`

> baseline 的意义：**给深度模型一个“必须超过”的下限**，也用于证明“不是数据/标签问题”。

---

### 4.2 BiLSTM（Pooling 版本）
```bash
make train-lstm
```

输出在：
- `outputs/bilstm_pool/best.pt`（按 val_f1 最优）
- `outputs/bilstm_pool/vocab.json`
- `outputs/bilstm_pool/summary.json`
- `outputs/bilstm_pool/history.json`
- `outputs/bilstm_pool/*report.txt`

---

### 4.3 BiLSTM + Attention
```bash
make train-attn
```

输出在：
- `outputs/bilstm_attention/best.pt`
- `outputs/bilstm_attention/vocab.json`
- 其他同上

---

## 5. 消融实验（ablation）

目标：回答“Attention 到底有没有用？”以及“Pooling 怎么选？”

```bash
make ablation
```

输出：
- `outputs/ablation/ablation_results.csv`
- 每个实验一个子目录 `outputs/ablation/<exp_name>/`

你可以直接在 `configs/ablation.yaml` 增删实验：
- pooling: last / mean / max
- attn_dim: 64 / 128 / 256
- 你也可以扩展更多维度（dropout、max_len、min_freq、bidirectional 等）

---

## 6. 错误分析（error analysis）

错误分析不是“看几个错例”，而是要做到：

1) 有全量预测表（每条样本：真值/预测/置信度/是否错）  
2) 有分桶（high-conf、borderline、长文本截断、unk 过高）  
3) 导出一份可人工标注的 sheet（error_type + comment），为下一轮迭代服务  

执行：

```bash
make error-analysis
```

输出：
- `outputs/error_analysis/test_predictions.csv`
- `outputs/error_analysis/test_predictions_with_diag.csv`
- `outputs/error_analysis/confusion_matrix.png`
- `outputs/error_analysis/samples_to_label.csv`（可人工标注）

---

## 8. 项目故事线

> **1 baseline → 改进 → ablation → error analysis**

- 数据：说明数据来源、清洗规则、划分方式（是否按 source_file 防泄漏）  
- baseline：TF-IDF + LR，指标是多少（给一个下限）  
- 深度模型：BiLSTM pooling，为什么 pooling（长文本、多位置关键信息）  
- 改进：Attention 让模型自动学习“关键词权重”，而不是固定 pooling  
- 消融：证明提升来自 attention，而不是别的（同数据同训练设置）  
- 错误分析：模型在哪些结构上错（否定、转折、长文本截断、词表覆盖不足）→ 下一步怎么改（更强 tokenizer / subword / 预训练模型 / 增强数据等）

---

## 9. 常见工程误区

- **notebook 混成一坨**：训练、数据、可视化、调参全在一个文件 → 无法复现  
- **没有配置文件**：每次改超参都要改代码 → 无法追溯  
- **没有输出规范**：模型/词表/指标散落各处 → 无法复盘  
- **数据泄漏**：同一 source_file 的段落同时出现在 train/test → 虚高指标  
- **不固定 seed**：跑一次一个结果 → 无法解释 ablation

---

## 10. FAQ

### Q: 我想把 LSTM 跑在自己的中文数据上？
把 `configs/lstm.yaml` 里的 dataset 改成：

```yaml
dataset:
  type: csv_splits
  split_dir: data/splits
  text_col: text
  label_col: label
  lang: zh
```

同理，Attention 也一样。

### Q: 你为什么还保留 notebooks/original？
面试时你可以展示“从 notebook → 工程化”的过程，但真正可复现的是 `scripts/ + src/ + configs/`。

---

## License
MIT
