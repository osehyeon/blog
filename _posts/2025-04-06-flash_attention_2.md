---
layout: single
title:  "[논문 리뷰] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
categories: "논문"
toc: true
typora-root-url: .
---

"FlashAttention-2" 논문은 프린스턴 대학교 박사 과정 학생 Tri Dao가 제안한 연구로, 트랜스포머 모델에서 긴 시퀀스를 처리할 때 셀프 어텐션 메커니즘이 가지는 메모리 및 계산 비효율성 문제를 극복하고자 개발되었습니다. 이 논문은 2023년 arXiv에 처음 공개되었으며, 이후 ICLR 2024에 포스터 형식으로 채택되어 발표되었습니다. 

## Standard Attetnion Implementation

$N$은 시퀀스 길이, $d$는 헤드 일 때, 입력 시퀀스 $\mathbf{Q, K, V}\in \mathbb{R}^{N \times d }$ 가 주어지면, 어텐션 출력인 $\mathbf{O} \in \mathbb{R}^{N \times d}$ 를 계산하는 것이 목표입니다. 

$$
\begin{align*}
\mathbf{S} &= \mathbf{QK}^T \in \mathbb{R}^{N \times N} \\
\mathbf{P} &= \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N} \\
\mathbf{O} &= \mathbf{PV} \in \mathbb{R}^{N \times d}
\end{align*}
$$
표준 어텐션은 행렬 $S$ 와 $P$ 를 HBM에 저장하는데, 이는 $O(N^2)$의 메모리가 사용됩니다. 보통 $N \gg d$ 인 상황에서 ($N$은 보통 1k-8k의 범위이며, $d$는 대략 64-128) 표준 어텐션의 구현은 다음과 같습니다. 

1. GEMM을 호출하여 $\mathbf{S = QK}^T$ 연산하고, 결과를 HBM에 저장합니다. 
2. HBM에서 $\mathbf{S}$를 로드하여 softmax를 계산하고, 결과 $\mathbf{P}$ 를 HBM에 저장합니다. 
3. GEMM을 호출하여 $\mathbf{O = PV}$ 연산하고, 어텐션 출력을 저장합니다.

## FlashAttention 

FlashAttention은 메모리 I/O를 줄이기 위해 타일링의 고전적인 기법을 적용합니다. 

1. HBM에서 SRAM으로 입력 블록을 로드합니다. 
2. 해당 블록에 대해 Attention 계산을 수행합니다.
3. 중간 결과 행렬인 $\mathbf{S}$ 와 $\mathbf{P}$를 HBM에 저장하지 않고, 즉시 출력 값을 저장합니다. 

여기서 softmax가 전체 행 정보가 필요로 하므로, online softmax기법을 사용하여, 어텐션 계산을 블록 단위로 분할하되, 각 블록의 출력을 재조정함으로, 전체 결과가 근사 없이 정확하게 나오도록 보장합니다. 

$$
\newcommand{\sn}[1]{^{(#1)}}
\newcommand{\m}{\boldsymbol{m}} 
\newcommand{\B}{\boldsymbol{B}} 
\newcommand{\P}{\mathbf{P}} 
\newcommand{\S}{\mathbf{S}} 
\newcommand{\Q}{\mathbf{Q}} 
\newcommand{\K}{\mathbf{K}} 
\newcommand{\V}{\mathbf{V}} 
\newcommand{\R}{\mathbb{R}}
\newcommand{\O}{\mathbf{O}} 

\begin{align*}
m^{(1)} &= \text{rowmax}(\S^{(1)}) \in \R^{\B_r} \\
\ell^{(1)} &= \text{rowsum}(e^{\s\sn{1}-\m\sn{1}}) \in \R^{\B_r} \\
\tilde{\P}\sn{1} &= \text{diag}(\ell\sn{1})^{-1}e^{\S\sn{1}-\m\sn{1}} \in \R^{\B_r \times \B_c} \\\
\O\sn{1} &= \tilde{\P}\sn{1}\V\sn{1} = \text{diag}(\ell\sn{1})^{-1} e^{\S\sn{1} - \m\sn{1}}\V\sn{1} \in \R^{\B_r \times d} \\
\m\sn{2} &= \max(\m\sn{1}, \text{rowmax}(\S\sn{2})) = \m \\
\ell\sn{2} &= e^{\m\sn{1} - \m\sn{2}} \ell\sn{1} + \text{rowsum}(e^{\S\sn{2}-\m\sn{2}}) = \text{rowsum}(e^{\S\sn{1}-\m}) + \text{rowsum}(e^{\S\sn{2}-\m})=\ell \\
\tilde{\P\sn{2}} &= \text{diag}(\ell\sn{2})^{-1}e^{\S\sn{2}-\m\sn{2}} \\ 
\O\sn{2} &= \text{diag}(\ell\sn{1} / \ell\sn{2})^{-1} \O\sn{1} + \tilde{\P\sn{2}}\V\sn{2} = \text{diag}(\ell\sn{2})^{-1}e^{\S\sn{1}-\m}\V\sn{1} + \text{diag}(\ell\sn{2})^{-1}e^{\S\sn{2}-\m}\V\sn{2} = \O
\end{align*}
$$

## FlashAttention-2

