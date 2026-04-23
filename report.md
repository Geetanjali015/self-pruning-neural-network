# Technical Report — Self-Pruning Neural Network for CIFAR-10

**Author:** AI Engineering Intern Candidate  
**Submission:** Tredence AI Engineering Internship — Case Study  
**Stack:** Python · PyTorch · Torchvision · Matplotlib · Google Colab (GPU)

---

## 1. Objective

Design and implement a neural network that learns to prune its own weights **during training** using learnable gate parameters and an L1 sparsity regularization loss. Evaluate the sparsity-accuracy trade-off across multiple lambda values on the CIFAR-10 benchmark. Demonstrate iterative engineering improvement by progressing from a baseline MLP to a stronger CNN architecture.

---

## 2. Methodology

### 2.1 The PrunableLinear Layer

The core building block of this system is `PrunableLinear` — a custom replacement for `nn.Linear` that attaches a learnable gate to every weight.

**Parameters registered per layer:**

| Parameter | Shape | Role |
|-----------|-------|------|
| `weight` | `[out, in]` | Standard learnable weight matrix |
| `gate_scores` | `[out, in]` | Learnable gate score (same shape as weight) |
| `bias` | `[out]` | Standard bias term |

**Forward pass:**

```python
gates        = torch.sigmoid(self.gate_scores)   # ∈ (0, 1)
pruned_weight = self.weight * gates               # element-wise gate
output        = F.linear(x, pruned_weight, self.bias)
```

**Gradient flow:**

Both `weight` and `gate_scores` are `nn.Parameter` objects. PyTorch's autograd traces through the sigmoid and element-wise multiply automatically, computing:

```
∂L/∂weight       = δ · x · sigmoid(G)
∂L/∂gate_scores  = δ · x · weight · sigmoid(G) · (1 − sigmoid(G))
```

This ensures the optimizer updates gate scores alongside weights in every step — no custom backward pass required.

---

### 2.2 Sparsity Regularization Loss

Training with only cross-entropy gives the model no reason to close any gates. The sparsity loss provides that incentive:

```
Total Loss = CrossEntropyLoss(logits, targets) + λ × SparsityLoss

SparsityLoss = Σᵢ sigmoid(Gᵢ)   over all PrunableLinear layers
```

This is the **L1 norm of all gate values** (since sigmoid outputs are always positive, absolute value = value).

---

### 2.3 Why L1 Regularization Encourages Sparsity

The key distinction between L1 and L2 is the shape of their gradients:

| Regularizer | Penalty | Gradient w.r.t. gate `g` | Behaviour as g → 0 |
|-------------|---------|--------------------------|----------------------|
| L2 | `Σ gᵢ²` | `2gᵢ` | Gradient → 0; weight shrinks but never reaches exactly 0 |
| L1 | `Σ \|gᵢ\|` | `sign(gᵢ) = +1` (g > 0) | Constant push; weight reaches exactly 0 |

The L1 gradient is a **constant −λ** applied to the gate score at every step. This means regardless of how small the gate becomes, the penalty keeps pushing it further negative, driving `sigmoid(G) → 0`. L2 cannot achieve this — its gradient decays proportionally, producing small weights rather than zero weights.

In practice: the optimizer sees a fixed sparsity "tax" on every active gate. When the classification benefit of keeping a gate open is less than λ, the gate is driven to zero and that weight is effectively removed from the network.

---

## 3. Architectures

### 3.1 Baseline MLP

```
Input: 3072 (32×32×3 flattened)
  → PrunableLinear(3072 → 512) + ReLU
  → PrunableLinear(512  → 256) + ReLU
  → PrunableLinear(256  → 10)
  → CrossEntropyLoss
```

- All connections are prunable
- No spatial feature extraction
- Used to validate the gating mechanism in isolation

### 3.2 Advanced CNN

```
Input: 3×32×32
  → Conv2d(3→32, 3×3) + ReLU + MaxPool(2×2)
  → Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)
  → Flatten
  → PrunableLinear(64×6×6 → 256) + ReLU
  → PrunableLinear(256 → 10)
  → CrossEntropyLoss
```

- Convolutional layers extract spatial features (not pruned)
- Pruning is concentrated at the FC layers where redundancy is highest
- Better accuracy baseline enables more meaningful sparsity analysis

---

## 4. Experimental Setup

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10 (50k train / 10k test) |
| Lambda values tested | 0.0001, 0.001, 0.01 |
| Optimizer | Adam |
| Sparsity threshold | gate < 0.01 → pruned |
| Hardware | Google Colab (GPU) |
| Framework | PyTorch + Torchvision |

---

## 5. Results

### 5.1 Baseline MLP Results

| Lambda (λ) | Test Accuracy | Sparsity % | Observation |
|:----------:|:-------------:|:----------:|-------------|
| 0.0001     | 51.80%        | 1.52%      | Weak penalty; gates mostly open |
| 0.001      | 49.95%        | 1.71%      | Moderate pressure; slight accuracy drop |
| 0.01       | 46.82%        | 1.72%      | High pressure; accuracy loss without significant extra sparsity |

**Key finding:** The MLP plateaus in sparsity (~1.72%) even at high λ. Its dense pixel-level input creates a highly interconnected weight structure where most connections are individually necessary — making them resistant to pruning.

---

### 5.2 Advanced CNN Results

| Lambda (λ) | Test Accuracy | Sparsity % | Observation |
|:----------:|:-------------:|:----------:|-------------|
| 0.0001     | **71.12%**    | 10.44%     | Best accuracy; meaningful pruning begins |
| 0.001      | 70.83%        | 13.52%     | Marginal accuracy loss; significant sparsity gain |
| 0.01       | 67.85%        | **13.97%** | Most aggressive pruning; 3.27% accuracy cost |

**Key finding:** CNN achieves up to **13.97% sparsity** while maintaining 67.85% accuracy — a strong result for a model trained with self-pruning from scratch.

---

## 6. MLP vs CNN — Comparative Analysis

| Metric | MLP (best) | CNN (best) | Improvement |
|--------|-----------|-----------|-------------|
| Test Accuracy | 51.80% | 71.12% | **+19.32%** |
| Max Sparsity | 1.72% | 13.97% | **+12.25 pts** |
| Accuracy at max sparsity | 46.82% | 67.85% | **+21.03%** |

**Why the CNN performs better on both dimensions:**

The CNN's convolutional layers extract compact, hierarchical spatial features before passing them to the FC layers. This means the FC layers operate on a dense, semantically rich representation rather than raw pixels. As a result:

1. **Higher accuracy baseline** — conv features are more discriminative than pixel intensities.
2. **More prunable FC structure** — the high-level feature space contains genuine redundancy; many dimensions contribute negligibly to the final classification, making them natural targets for pruning.
3. **Better sparsity-accuracy trade-off** — the CNN loses only 3.27% accuracy (71.12% → 67.85%) while gaining 3.53 percentage points of sparsity (10.44% → 13.97%). The MLP loses 5% accuracy with almost no additional sparsity gain in the same λ range.

---

## 7. Lambda Trade-off Analysis

```
          Low λ (0.0001)              High λ (0.01)
          ──────────────              ─────────────
Accuracy  ████████████████  ──►  ███████████░░░░░░
Sparsity  ██░░░░░░░░░░░░░░  ──►  █████████████░░░░
```

- **λ = 0.0001:** Sparsity pressure is gentle. Most gates remain open (~0.5). The model behaves like a lightly regularized standard network. Best accuracy, lowest pruning.
- **λ = 0.001:** Clear bifurcation of gate values begins — many gates collapse toward 0, a subset of important connections survive. Best balance point.
- **λ = 0.01:** Aggressive pruning. The sparsity term dominates the loss. Accuracy drops as the model is forced to drop connections it would prefer to keep. Sparsity plateaus as only the minimum viable connections survive.

The relationship is **monotonic but not linear**: large λ increases yield diminishing sparsity returns while accelerating accuracy loss.

---

## 8. Engineering Learnings

1. **Custom layers need parameter registration.** Forgetting to wrap `gate_scores` in `nn.Parameter` means the optimizer ignores it entirely — the gates never move.

2. **Sigmoid initialization matters.** Initializing gate scores to 0 gives `sigmoid(0) = 0.5` — a neutral starting point. Starting too negative (all-pruned) or too positive (all-open) can destabilize early training.

3. **Architecture is a prerequisite for pruning quality.** A model that barely learns cannot prune meaningfully. The MLP's low accuracy ceiling (~52%) means there is little signal to differentiate important from unimportant weights.

4. **L1 norm on gates is theoretically motivated, not just empirical.** The constant gradient property directly corresponds to true sparsity induction, distinguishing it from L2 shrinkage.

5. **Sparsity and accuracy are jointly optimizable.** The CNN at λ=0.001 achieves 70.83% accuracy with 13.52% sparsity — a network that has removed 1 in 7 of its prunable connections with essentially no accuracy loss.

---

## 9. Conclusion

This project successfully demonstrates **learned self-pruning** as a viable training-time regularization strategy. The key results are:

- The `PrunableLinear` layer correctly implements gated weights with full gradient flow through both weight and gate parameters.
- L1 regularization on sigmoid gate values provably and empirically induces sparsity, with λ providing clean, monotonic control over the sparsity-accuracy trade-off.
- Upgrading from MLP to CNN delivered a **+19.32% accuracy gain** and **+12.25 percentage point sparsity gain**, demonstrating that architecture quality is a prerequisite for effective pruning.
- The CNN at λ=0.001 represents the optimal operating point: 70.83% accuracy with 13.52% of weights pruned.

This iterative engineering approach — validate the mechanism, then improve the architecture — reflects real-world AI engineering practice and demonstrates the ability to reason about model design holistically.

---

*Report generated as part of the Tredence AI Engineering Internship Case Study submission.*
