# LightGCN vs NGCF for Recommendation

<p align="center">
  <strong>Mini Project - Data Mining</strong><br>
  Comparative study and reproduction of <strong>NGCF</strong> and <strong>LightGCN</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-orange">
  <img alt="Task" src="https://img.shields.io/badge/Task-Recommendation-success">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-Amazon--book%20%7C%20Amazon--software--2023-informational">
</p>

---

## 📌 Overview
This repository contains a mini project for studying and comparing two graph-based recommendation models:
- **NGCF** (*Neural Graph Collaborative Filtering*)
- **LightGCN** (*Light Graph Convolution Network*)

The project focuses on:
- understanding the theoretical differences between NGCF and LightGCN,
- reproducing experiments on **Amazon-book**,
- evaluating both models on a new dataset, **Amazon-software-2023**,
- comparing ranking quality using **Recall@20** and **NDCG@20**.

---

## 🎯 Objectives
- Explain why recommendation can be modeled on a **user-item bipartite graph** instead of only using matrix factorization.
- Analyze the architectural differences between **NGCF** and **LightGCN**.
- Reproduce benchmark results on **Amazon-book**.
- Test generalization on **Amazon-software-2023**.
- Compare simplicity vs. effectiveness in graph-based collaborative filtering.

---

## 🧠 Model Summary
### NGCF
NGCF extends GCN for recommendation by combining:
- neighborhood aggregation,
- feature transformation,
- nonlinear activation,
- multi-layer embedding aggregation.

### LightGCN
LightGCN simplifies NGCF by removing:
- feature transformation,
- nonlinear activation,
- explicit self-connection,

and keeping only:
- **normalized neighborhood aggregation**,
- **layer combination**.

This design makes LightGCN simpler, easier to train, and often more effective for **ID-based collaborative filtering**.  

---

## 🗂️ Project Structure
```text
.
├── NGCF.ipynb
├── NGCF-dataset_software.ipynb
├── LightGCN.ipynb
├── LightGCN-dataset_software.ipynb
├── NGCF.py
├── main.py
└── ../requirements.txt
```

### Notebook Roles
- **NGCF.ipynb**: Train/evaluate NGCF on `amazon-book`
- **NGCF-dataset_software.ipynb**: Train/evaluate NGCF on `amazon-software-2023`
- **LightGCN.ipynb**: Train/evaluate LightGCN on `amazon-book`
- **LightGCN-dataset_software.ipynb**: Train/evaluate LightGCN on `amazon-software-2023`

---

## ⚙️ Environment
This project uses **two different deep learning frameworks**:

### For NGCF
- Python 3.x
- Jupyter Notebook
- TensorFlow (`tf-nightly`)
- scikit-learn
- scipy

### For LightGCN
- Python 3.x
- Jupyter Notebook
- PyTorch
- CUDA-enabled GPU (optional but recommended)

> **Recommendation:** use separate virtual environments for NGCF and LightGCN if dependency conflicts occur.

---

## 🚀 Quick Start
### 1) Install dependencies
```bash
pip install -r ../requirements.txt
```

### 2) Additional setup for NGCF
```bash
python -m pip install -U pip setuptools wheel
python -m pip install --pre -U tf-nightly
python -m pip install -U scikit-learn scipy
```

### 3) Check GPU availability
#### TensorFlow
```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

#### PyTorch
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

## ▶️ Run Experiments

## NGCF on Amazon-book
```bash
python NGCF.py \
  --dataset amazon-book \
  --regs [2e-5] \
  --embed_size 64 \
  --layer_size [64,64,64] \
  --lr 0.0005 \
  --save_flag 1 \
  --pretrain 0 \
  --batch_size 1024 \
  --epoch 200 \
  --verbose 50 \
  --node_dropout [0.1] \
  --mess_dropout [0.1,0.1,0.1]
```

## NGCF on Amazon-software-2023
```bash
python NGCF.py \
  --dataset amazon-software-2023 \
  --regs [2e-5] \
  --embed_size 64 \
  --layer_size [64,64,64] \
  --lr 0.001 \
  --save_flag 1 \
  --pretrain 0 \
  --batch_size 1024 \
  --epoch 200 \
  --verbose 50 \
  --node_dropout [0.1] \
  --mess_dropout [0.1,0.1,0.1]
```

## LightGCN on Amazon-book
```bash
python main.py \
  --decay=2e-4 \
  --lr=0.001 \
  --layer=3 \
  --seed=2020 \
  --dataset="amazon-book" \
  --topks="[20]" \
  --recdim=64
```

## LightGCN on Amazon-software-2023
```bash
python main.py \
  --decay=2e-4 \
  --lr=0.001 \
  --layer=3 \
  --seed=2020 \
  --dataset="amazon-software-2023" \
  --topks="[20]" \
  --recdim=64
```

---

## 📊 Datasets
### Amazon-book
Main benchmark dataset used for reproduction.  
- Users: **52,643**  
- Items: **91,599**  
- Interactions: **2,984,108**  
- Density: **0.0619%**

### Amazon-software-2023
New dataset used to test generalization.  
- Users: **29,809**  
- Items: **6,346**  
- Interactions: **491,597**  
- Density: **0.26%**

---

## 🧪 Experimental Settings
### NGCF
- Embedding size: `64`
- Number of layers: `3`
- Batch size: `1024`
- Epochs: `200`
- Node dropout: `0.1`
- Message dropout: `[0.1, 0.1, 0.1]`

### LightGCN
- Embedding size (`recdim`): `64`
- Number of layers: `3`
- Weight decay: `2e-4`
- Seed: `2020`
- Top-K: `20`

---

## 📈 Results
### Reproduction on Amazon-book
| Model | Recall@20 (Paper) | Recall@20 (Reproduced) | NDCG@20 (Paper) | NDCG@20 (Reproduced) |
|---|---:|---:|---:|---:|
| NGCF | 0.0337 | 0.03112 | 0.0261 | 0.05899 |
| LightGCN | 0.0410 | 0.0405 | 0.0318 | 0.0312 |

### Training Time
| Model | Reported Runtime |
|---|---|
| NGCF | 24 hours (90 epochs) |
| LightGCN | 24 hours (1000 epochs) |

### Results on Amazon-software-2023
| Metric@20 | NGCF | LightGCN |
|---|---:|---:|
| Recall | 0.24364 | 0.275 |
| NDCG | 0.21250 | 0.181 |

### Key Observations
- On **Amazon-book**, LightGCN reproduces paper results more closely in **Recall@20**.
- On **Amazon-software-2023**, LightGCN achieves **higher Recall**, but NGCF achieves **higher NDCG**.
- This suggests that LightGCN is strong at retrieving relevant items into the top-K list, while NGCF may rank the relevant items more favorably in some cases.

---

## ✅ Advantages of LightGCN
- Simpler architecture than NGCF
- Fewer trainable parameters
- Easier to reproduce and deploy
- Effective at leveraging high-order connectivity in user-item graphs

## ⚠️ Limitations
- Risk of **over-smoothing** when stacking too many layers
- Relies only on ID embeddings
- Does not directly incorporate richer item/user features such as text or images

---

## 🔍 Notes
- Make sure the relative path `../requirements.txt` is correct before running the notebooks.
- Confirm that dataset names exactly match the command-line arguments:
  - `amazon-book`
  - `amazon-software-2023`
- For faster training, verify GPU access before running experiments.
- NGCF currently uses `tf-nightly`, so exact reproducibility may vary depending on installation date/version.
- In the notebooks, **LightGCN uses `--decay=2e-4`**, which should be treated as the actual experimental setting.

---

## 📚 References
1. Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang. **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation**. arXiv:2002.02126.
2. Xiang Wang et al. **Neural Graph Collaborative Filtering**. SIGIR 2019.

---

## 👥 Team Information
- **Course**: Data Mining
- **Supervisor**: PGS.TS Trần Minh Quang
- **Team 10**:
  - Trương Huy Thái
  - Văn Phi Cảnh
  - Vương Minh Toàn

---

## 📌 Conclusion
This project highlights an important idea in recommendation system design:

> **A simpler model can still outperform a more complex one if it preserves the truly essential components.**

LightGCN demonstrates that for ID-based collaborative filtering, keeping only **neighborhood aggregation** and **layer combination** can be sufficient to achieve strong recommendation performance.
