

# 🧠 Self-Pruning Neural Network for CIFAR-10
### Tredence AI Engineering Intern — Case Study Submission
> A PyTorch implementation of neural networks that learn to prune their own weights **during training** — not after — using learnable sigmoid gates and L1 sparsity regularization.

---

## 📌 Problem Statement

Standard neural network pruning is a **post-training** process: train a large model, then remove unimportant weights. This project explores a fundamentally different approach — embedding the pruning mechanism directly into the learning process itself.

Each weight in the network is paired with a learnable **gate parameter**. During training, the model simultaneously learns *what* to represent and *which connections to discard*, guided by a sparsity penalty in the loss function.

---

## 🔬 Core Methodology

### Gated Weight Mechanism

For every weight matrix `W` in a prunable layer, a corresponding gate score matrix `G` of the same shape is registered as a learnable parameter.

```
gate    = sigmoid(G)           ∈ (0, 1)
W_eff   = W × gate             (element-wise)
output  = W_eff · x + bias
```

- `gate ≈ 1` → connection is preserved  
- `gate ≈ 0` → connection is suppressed / pruned  

### Loss Formulation

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss

SparsityLoss = Σ sigmoid(Gᵢ)   (L1 norm of all gate values)
```

`λ` (lambda) is a hyperparameter controlling the sparsity-accuracy trade-off. A higher λ pushes more gates toward zero, increasing pruning at the cost of model capacity.

### Why L1 Encourages Sparsity

The L1 penalty applies a **constant gradient** of magnitude λ against every gate, regardless of its current value. Unlike L2 (whose gradient shrinks to zero near zero), L1 keeps pushing gate scores toward −∞, causing `sigmoid(G)` to collapse to **exactly 0** — fully removing that weight from the computation graph.

---

## 🏗️ Models Implemented

### 1. Baseline MLP — `PrunableLinear` Validation

A simple multi-layer perceptron to verify the gating mechanism works correctly before scaling up.

```
Input (3072)
  └─► PrunableLinear(3072 → 512) + ReLU
        └─► PrunableLinear(512 → 256) + ReLU
              └─► PrunableLinear(256 → 10)
                    └─► Logits
```

**Purpose:** Isolate and validate the PrunableLinear layer. Confirm gradients flow through both `W` and `G`. Establish a baseline for comparison.

---

### 2. Advanced CNN — Production-Grade Model

A convolutional architecture that combines spatial feature extraction with prunable fully-connected layers.

```
Input (3×32×32)
  └─► Conv(3→32) + ReLU + MaxPool
        └─► Conv(32→64) + ReLU + MaxPool
              └─► Flatten
                    └─► PrunableLinear(→ 256) + ReLU
                          └─► PrunableLinear(→ 10)
                                └─► Logits
```

**Purpose:** Leverage convolutional inductive biases for CIFAR-10's spatial structure. Apply pruning only at the FC layers where redundancy is highest.

---

## 📊 Results

### Baseline MLP

| Lambda (λ) | Test Accuracy | Sparsity % |
|:----------:|:-------------:|:----------:|
| 0.0001     | 51.80%        | 1.52%      |
| 0.001      | 49.95%        | 1.71%      |
| 0.01       | 46.82%        | 1.72%      |

### Advanced CNN

| Lambda (λ) | Test Accuracy | Sparsity % |
|:----------:|:-------------:|:----------:|
| 0.0001     | **71.12%**    | 10.44%     |
| 0.001      | 70.83%        | 13.52%     |
| 0.01       | 67.85%        | **13.97%** |

---

## 📈 Why CNN Outperformed MLP

| Factor | MLP | CNN |
|--------|-----|-----|
| Feature extraction | Raw pixel values (3072-dim) | Hierarchical spatial features |
| Parameter efficiency | Low — dense fully-connected | High — shared conv filters |
| Translation invariance | ✗ | ✓ |
| Pruning effectiveness | 1.52–1.72% | 10.44–13.97% |
| Best accuracy | 51.80% | 71.12% |

The MLP treats every pixel independently, creating a dense, hard-to-prune weight landscape. The CNN builds spatially aware representations first, leaving the FC layers — where PrunableLinear operates — with more structured and prunable redundancy.

---

## 🗂️ Folder Structure

```
self-pruning-nn/
│
├── self_pruning_network.py     # Core implementation (PrunableLinear, MLP, CNN, training)
├── report.md                   # Technical report with full analysis
├── README.md                   # This file
│
├── data/                       # CIFAR-10 (auto-downloaded)
│
└── outputs/
    ├── results.csv             # MLP experiment results
    ├── cnn_results.csv         # CNN experiment results
    ├── gate_dist_*.png         # Gate distribution histograms per λ
    ├── accuracy_curves.png     # Accuracy over epochs
    ├── sparsity_curves.png     # Sparsity over epochs
    └── best_model.pth          # Saved best model checkpoint
```

---

## ⚡ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/self-pruning-nn.git
cd self-pruning-nn

# 2. Install dependencies
pip install torch torchvision matplotlib numpy pandas

# 3. Run training (CIFAR-10 downloads automatically)
python self_pruning_network.py
```

> Recommended: Run on **Google Colab with GPU** for fastest results (~5–10 min).  
> CPU runtime: ~30–50 min for full experiment.

---

## Screenshots
### 1. Baseline MLP 
<img width="693" height="442" alt="image" src="https://github.com/user-attachments/assets/4a5a52dc-a2bf-4509-a6f2-4848cbb95cb0" />

### 2. CNN
<img width="720" height="466" alt="image" src="https://github.com/user-attachments/assets/23ff711e-8cb7-415a-8bb1-49240c311adb" />



## 💡 Key Learnings

- **L1 regularization on sigmoid gates** is a principled and effective mechanism for inducing true weight sparsity during training.
- **Architecture matters for pruning:** CNNs naturally concentrate learnable redundancy in FC layers, making them better pruning candidates than flat MLPs.
- **λ is a knob, not a switch:** Small changes in λ produce measurable and predictable changes in both accuracy and sparsity — evidence that the mechanism is working correctly.
- **Gradient flow through custom layers** requires careful design; registering gate scores as `nn.Parameter` ensures the optimizer updates them alongside weights.

---

## 🚀 Future Improvements

- Apply pruning directly to convolutional filters (structured pruning)
- Implement hard gating with a straight-through estimator for binary gates
- Explore learned λ scheduling (warm-up then anneal sparsity pressure)
- Benchmark inference speedup after zeroing pruned weights
- Compare against magnitude-based post-training pruning baselines

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Torchvision` · `NumPy` · `Pandas` · `Matplotlib` · `Google Colab (GPU)`

---

## 📄 License

MIT License — free to use, modify, and distribute.
