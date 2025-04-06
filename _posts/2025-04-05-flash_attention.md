---
layout: single
title:  "[논문 리뷰] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
categories: "논문"
toc: true
typora-root-url: .
---

"FlashAttention" 논문은 스탠포드 대학교 박사 과정 학생인 Tri Dao가 Transformer 모델에서 긴 시퀀스를 처리할 때 self-attention 메커니즘이 메모리와 계산 효율성 면에서 비효율적인 문제를 해결하고자 제안되었습니다. 해당 연구는 2022년 5월에 arXiv에 처음 공개되었으며, 이후 NeurIPS 2022(Advances in Neural Information Processing Systems)에서 포스터 발표 형식으로 소개되었습니다.

![image-20250405182831999](../../images/2025-04-05-int_flashatttention/image-20250405182831999.png)

해당 이미지는 FlashAttention의 핵심 아이디어를 설명합니다.

### 메모리 계층 구조 

GPU는 **SRAM**, **HBM**(High Bandwidth Memory), CPU **DRAM**으로 구성된 메모리 계층 구조를 가집니다. FlashAttention은 이러한 계층 구조를 고려하여 메모리 접근을 최적화합니다.

### 작동 방식 

**Tiling** : FlashAttention은 입력(Q, K, V)를 블록으로 나누어 **SRAM**에 불러오고, 블록단위로 Attention을 계산합니다. 

**Outer Loop** : K와 V 행렬의 블록을 순회하면서 **SRAM**에 불러옵니다. 

**Inner Loop** : Q 행렬의 블록을 순회하여 **SRAM**에 로드하고 Attention 계산 결과를 **HBM** 에 저장합니다.

여기서 핵심은 큰 Attention 행렬을 **HBM**에 저장하지 않고, 필요할 때마다 **SRAM**에서 계산하여 메모리 접근을 줄이는 것입니다.

### 연산 속도 비교 

Pytorch는 Matmul, Mask, Softmax, Dropout 등의 연산을 순차적으로 수행합니다. 

FlashAttention은 이러한 연산들을 Fused Kernel로 통합하여 처리합니다.

![Flash Attention](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/flash-attn.png)

## Standard Attention Implementation

입력 시퀀스 $Q, K, V \in \mathbb{R}^{N \times d}$ 에서 $N$은 시퀀스 길이이고, $d$ 는 헤드 차원이입니다. 최종적으로 $O \in R^{N \times d}$ 를 계산하는 것이 목표입니다.

$$
\begin{align*}

\mathbf{S} &= \mathbf{QK}^T \in \mathbb{R}^{N \times N} \\
\mathbf{P} &= \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N} \\
\mathbf{O} &= \mathbf{PV} \in \mathbb{R}^{N \times d}

\end{align*}
$$

| (Algorithm 0)  Standard Attention Implementation             |
| ------------------------------------------------------------ |
| **Require** : Matrces $Q, K, V \in \mathbb{R}^{N \times d} $ in HBM. |
| 1: Load $\mathbf{Q, K}$ by blocks from HBM, compute $\mathbf{S = QK^T}$, writh $\mathbf{S}$ to HBM |
| 2: Read $\mathbf{S}$ from HBM, compute $\mathbf{P} = \text{softmax}(\mathbf{S})$, write $\mathbf{P}$ to HDM |
| 3: Load $\mathbf{P}$ and $\mathbf{V}$  by blocks from HBM, compute $\mathbf{O = PV}$, writhe $\mathbf{O}$ to HBM |

## FlashAttention

주어진 입력 $Q, K, V \in \mathbb{R}^{N \times d}$ 를 HBM에서 가져와, 어텐션 출력 $O \in \mathbb{R}^{N \times d}$ 를 계산하고 HBM에 기록하는 것을 목표로 합니다.

### Tilting 

벡터 $x \in \mathbb{R}^{B}$ 의 소프트맥스 계산은 다음과 같습니다. 

$$
\begin{align*}
m(x) &:= \underset{i}{\max} x_i \\
f(x) &:= [e^{x_1-m(x)} \dots e^{x_B-m(x)}] \\ 
\ell(x) &:= \sum_i f(x)_i \\ 
\text{softmax}(x) &:= \frac{f(x)}{\ell(x)}
\end{align*}
$$

벡터 $x^{(1)}, x^{(2)} \in \mathbb{R}^B$에 $x = [x^{(1)} \; x^{(2)} \in \mathbb{R}^{2B}]$ 로 연결된 소프트맥스를 다음과 같이 분해할 수 있습니다. 

$$
\begin{align*}
m(x) &= m([x^{(1)} \; x^{(2)}]) = \max(x^{(1)}\; x^{(2)}) \\
f(x) &= [e^{m(x^{(1)}) - m(x)} f(x^{(1)}) \quad  e^{m(x^{(2)}) - m(x)} f(x^{(2)})] \\ 
\ell(x) &= \ell([x^{(1)} \; x^{(2)}]) = e^{m(x^{(1)}) - m(x)} \ell(x^{(1)}) \quad  e^{m(x^{(2)}) - m(x)} \ell(x^{(2)}) \\
\text{softmax}(x) &= \frac{f(x)}{\ell(x)}
\end{align*}
$$

| (Algorithm 1) FlashAttention |
| --- |
| **Require:**  Matrices $\mathbf{Q, K, V} \in \mathbb{R}^{N \times d}$ in HBM, on-chip SRAM of size M. |
| 1: Set block sizes $B_c = \lceil \frac{M}{4d} \rceil, B_r = \min(\lceil \frac{M}{4d}, d \rceil )$. |
| 2: Initialize $\mathbf{O} = (0)_{N \times d} \in \mathbb{R}^{N \times d}, \ell = (0)_N \in \mathbb{R}^N, m = (-\infty)_N \in \mathbb{R}^N$ in HBM. |
| 3: Divide $\mathbf{Q}$ into $T_r = \lceil \frac{N}{B_r} \rceil$ blocks $\mathbf{Q_1, \dots, Q}_{T_r}$ of size $B_r \times d$  each, and divide $\mathbf{K, V}$ in to $T_c = \lceil \frac{N}{B_c} \rceil$  blocks. |
| 4: Divide $\mathbf{O}$ into $T_r$ blocks $\mathbf{O}\_1$, $\dots,$ $\mathbf{O}\_{T_r}$ of size $B_r \times d$ each, devide $\ell$ into $T_r$ blocks $\ell_1, \dots, \ell_{T_r}$ of size $B_r$ each, and devide $m$ into $T_r$ blocks $m_1, \dots, m_{T_r}$ of size $B_r$ each. |
| 5: $\text{for } 1 \leq j \leq T_c \text{ do}$ |
| 6: $\quad$ Load $\mathbf{K}_j, \mathbf{V}_j$ from HBM to on-chip SRAM |
| 7: $\quad$ $\text{for } 1 \leq i \leq T_r \text{ do}$ |
| 8: $\quad \ \ \quad$ Load $\mathbf{Q}_i, \mathbf{O}_i, \ell_i, m_i$ from HBM to on-chip SRAM |
| 9: $\quad \ \ \quad$ On chip, compute $\mathbf{S}_{ij}$ = $\mathbf{Q_i K_j^T} \in \mathbb{R}^{B_r \times B_c}$ |
| 10: $\quad \quad$ On chip, compute $\tilde{m}\_{ij} = \text{rowmax}(\mathbf{S}\_{ij}) \in \mathbb{R}^{B_r}$, $\tilde{\mathbf{P}}\_{ij} = e^{\mathbf{S}\_{ij} -\tilde{m}\_{ij}} \in \mathbb{R}^{B_r \times B_c} \text{ (pointwise)}$, $\text{resume}(\tilde{\mathbf{P}}_{ij}) \in \mathbb{R}^{B_r}$. |
| 11: $\quad \quad$ On chip, compute $m_i^{\text{new}} = \text{max}(m_i, \tilde{m_{ij}}) \in \mathbb{R}^{B_r}, \ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}}\ell_i + e^{\tilde{m}\_{ij} - m_i^{\text{new}}}\tilde{\ell}_{ij} \in \mathbb{R}^{B_r}$, |
| 12: $\quad \quad$ Write $\mathbf{O}\_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1} (\text{diag}(\ell_i)e^{m_i - m_i^{\text{new}}} \mathbf{O}\_i + e^{e^{\tilde{m}\_{ij}} - m_i^{\text{new}}}\tilde{\mathbf{P}}_{ij}\mathbf{V}_j)$ to HBM |
| 13: $\quad \quad$ Write $\ell_i \leftarrow \ell_i^{\text{new}}, m_i \leftarrow m_i^{\text{new}}$  to HBM |
| 14: $\quad$ $\text{end for}$ |
| 15: $\text{end for}$ |
| 16: Return $\mathbf{O}$. |
