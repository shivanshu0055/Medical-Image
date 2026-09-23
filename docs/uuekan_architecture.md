# UUEKAN: Edge-Enhanced Kolmogorov-Arnold Network with Uncertainty-Guided Attention

> **Paper Reference:**  
> Quanyi Chen, Haibin Wan, Xiuyu Yue, Xuejun Zhang.  
> *"UUEKAN: An edge-enhanced Kolmogorov-Arnold Network with uncertainty-guided attention for medical image segmentation."*  
> *Biomedical Signal Processing and Control*, Volume 124, 2026.  
> **Official Code:** [github.com/2369917438/UUEKAN](https://github.com/2369917438/UUEKAN)

---

## 1. Executive Summary & Motivation

In medical image segmentation (specifically Brain MRI tumor delineation in the **BRISC** benchmark), standard U-Net architectures encounter three fundamental bottlenecks:

1. **Fixed Activation Functions**: Standard CNNs (and Vision Transformers) rely on fixed node activation functions ($\text{ReLU}(x)$, $\text{GELU}(x)$). This static mapping lacks the adaptive flexibility to fit irregular, non-linear lesion contours without an explosion in parameter count.
2. **Attenuation of High-Frequency Boundary Signals in Deep Layers**: As feature maps are progressively downsampled into the bottleneck, the network learns semantic categories ("what") but loses topological boundary precision ("where it changes").
3. **Noisy Skip Connections**: Direct concatenation between shallow and deep layers transfers high-resolution spatial information contaminated by scanner noise and imaging artifacts.

**UUEKAN** solves these challenges by combining:
- **Kolmogorov-Arnold Networks (KAN)**: Deploying learnable B-spline activation functions on network edges rather than fixed activations on nodes.
- **EKAN (Edge-Enhanced KAN)**: Injecting shallow Sobel boundary priors directly into deep KAN bottleneck layers.
- **U-MALA (Uncertainty-guided Magnitude-Aware Linear Attention)**: Replacing basic skip connections with uncertainty-guided linear attention to filter noise and refine ambiguous boundaries.

---

## 2. Theoretical Foundations

### 2.1 MLP vs. Kolmogorov-Arnold Networks (KAN)
In a traditional Multi-Layer Perceptron (MLP), linear weight matrices $W$ alternate with fixed non-linear activations $\sigma$:
$$\text{MLP}(Z) = (W_{L-1} \circ \sigma \circ \dots \circ \sigma \circ W_0)(Z)$$

By contrast, the **Kolmogorov-Arnold representation theorem** states that any multivariate continuous function can be represented as a finite composition of continuous single-variable functions. A KAN layer parameterizes non-linear transformations on the connection **edges**:
$$\text{KAN}(Z)_j = \sum_{i=1}^{n_{in}} \phi_{j,i}(z_i)$$

Each 1D activation function $\phi(z)$ is parameterized as a linear combination of B-spline basis functions:
$$\phi(z) = w_b \cdot \text{SiLU}(z) + \sum_{k} c_k B_k(z)$$
where $B_k(z)$ are fixed B-spline basis functions, and $c_k$ are learnable coefficients. This allows each connection to evolve an adaptive curve ruler tailored to local high-frequency boundary textures.

---

## 3. UUEKAN Architecture Overview

```
Input MRI (3, 512, 512)
   │
   ├── Stage 1: Conv Block ────────────────────────┐
   │      │                                         │
   ├── Stage 2: Conv Block ──────────────────┐      │
   │      │                                  │      │
   ├── Stage 3: Conv Block ────────────┐     │      │
   │      │                            │     │      │
   ├── Stage 4: EKAN Block (Tokenized) │     │      │
   │      │                            │     │      │
   └── Stage 5: EKAN Bottleneck        │     │      │
          │                            │     │      │
       Upsample                        │     │      │
          │                            │     │      │
   ┌── Decoder Stage 4 (EKAN)          │     │      │
   │      │ + U-MALA Skip 4 ───────────┘     │      │
   ├── Decoder Stage 3 (Conv)                │      │
   │      │ + U-MALA Skip 3 ─────────────────┘      │
   ├── Decoder Stage 2 (Conv)                       │
   │      │ + U-MALA Skip 2 ────────────────────────┘
   ├── Decoder Stage 1 (Conv)
   │      │ + U-MALA Skip 1 (from t1)
   └── Final 1x1 Conv ──> Output Segmentation Mask (1, 512, 512)
```

### 3.1 Hybrid Division of Labor
* **Shallow Stages (1–3)**: Spline computations over high-resolution feature maps ($512 \times 512, 256 \times 256$) are computationally heavy. Standard convolutions are retained in stages 1–3 to extract low-level spatial features with minimal FLOPs.
* **Deep Stages (4–5)**: Features are downsampled to $32 \times 32$ and $16 \times 16$. Here, EKAN modules process tokenized sequences, leveraging KAN's expressive power where spatial dimensions are compact.

### 3.2 EKAN: Edge-Enhanced KAN
The shallowest feature map $t_1$ is routed through parallel horizontal and vertical Sobel filters:
$$S_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1\end{bmatrix}, \quad S_y = \begin{bmatrix}-1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1\end{bmatrix}$$
The resulting boundary maps are concatenated, projected via $1 \times 1$ convolution + BatchNorm, downsampled, and added to the main stream before entering the KANLayer. This keeps deep semantic features grounded in anatomical boundaries.

### 3.3 U-MALA: Uncertainty-Guided Magnitude-Aware Linear Attention
To bridge the semantic gap in skip connections:
1. **Uncertainty Map Generation (UMG)**:
   - Deep decoder features $F_D$ produce a probability map $P = \sigma(\text{Conv}_{1 \times 1}(F_D))$.
   - Uncertainty is computed based on distance from the decision boundary $\tau = 0.5$:
     $$U_{i,j} = \tau - |P_{i,j} - \tau|$$
   - Smoothed via a $7 \times 7$ Gaussian kernel ($\sigma=1.0$) and percentile-normalized to $[0, 1]$.
2. **Uncertainty-Aware Dynamic Partitioning (UDP)**:
   - A Quadtree dynamically inspects uncertainty density.
   - Low uncertainty $\rightarrow$ fine local quadrants.
   - High uncertainty $\rightarrow$ large receptive field for global context compensation.
3. **Magnitude-Aware Linear Attention (MALA)**:
   - Evaluates linear $O(N)$ attention without quadratic softmax:
     $$\text{Att}_m(Q_i, K_j) = \beta_i \cdot \phi(Q_i)\phi(K_j)^T - \gamma_i$$
   - $\beta_i = 1 + \frac{1}{z_i + \epsilon}$ and $\gamma_i = z_i$, amplifying weak boundary signals and preventing over-smoothing.

---

## 4. Multi-Stage Supervision & Tri-Partite Loss

During training, the total objective is:
$$L_{total} = L_{seg} + L_b + L_u$$

1. **Segmentation Loss ($L_{seg}$)**:
   - Summed over the main output and 4 intermediate decoder stages (Deep Supervision):
     $$L_{seg} = \sum_{i=1}^N w_i \left( 0.5 \cdot L_{BCE}(\hat{y}_i, y) + L_{Dice}(\hat{y}_i, y) \right)$$
2. **Boundary Loss ($L_b$)**:
   - Supervised using the Mean Squared Error between Sobel gradients:
     $$L_b = \|\text{Sobel}(\sigma(\hat{y})) - \text{Sobel}(y)\|^2$$
3. **Uncertainty Loss ($L_u$)**:
   - $L_{reg}$: Penalizes uncertainty in correct non-boundary regions while enforcing target uncertainty ($0.6$) at boundaries.
   - $L_{con}$: Consistency loss between upsampled deep uncertainty maps and shallower uncertainty maps.
   - $L_{thr}$: Regularization preventing dynamic partition thresholds from saturating.

---

## 5. Performance Comparison on BRISC (MRI)

From Table 1 and Table 2 in the published paper:

| Architecture | Paradigm | BRISC IoU (%) | BRISC Dice (%) | Params (M) | GFLOPs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | CNN | 86.27 ± 1.50 | 92.41 ± 0.87 | 31.04 | 218.95 |
| **U-Net++** | Dense CNN | 86.96 ± 0.94 | 93.02 ± 0.55 | 36.62 | 554.64 |
| **Att-Unet** | Attention CNN | 84.72 ± 0.92 | 91.59 ± 0.57 | 31.39 | 223.39 |
| **U-Mamba** | State-Space Model | 84.46 ± 0.27 | 91.58 ± 0.16 | 47.51 | 26.40 |
| **SegUKAN** | Pure KAN | 87.48 ± 0.80 | 93.31 ± 0.46 | 7.70 | 7.02 |
| **MedNeXt** | Scaled ConvNet | 90.93 ± 0.54 | 95.86 ± 0.28 | 42.35 | 289.47 |
| **UUEKAN (Ours)** | **Hybrid KAN + Edge** | **90.45 ± 0.27** | **94.79 ± 0.26** | **8.84** | **9.35** |

### Key Observations:
- **Accuracy**: UUEKAN beats standard U-Net by **+4.18% IoU** and **+2.38% Dice** on BRISC.
- **Efficiency**: UUEKAN achieves near-MedNeXt SOTA accuracy while using **79% fewer parameters** (8.84M vs 42.35M) and **96.8% fewer FLOPs** (9.35 vs 289.47 GFLOPs).
