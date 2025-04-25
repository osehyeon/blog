---
layout: single
title:  "[논문 리뷰] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
categories: "논문"
toc: true
typora-root-url: ./typora-root-url
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
\ell^{(1)} &= \text{rowsum}(e^{\S\sn{1}-\m\sn{1}}) \in \R^{\B_r} \\
\tilde{\P}\sn{1} &= \text{diag}(\ell\sn{1})^{-1}e^{\S\sn{1}-\m\sn{1}} \in \R^{\B_r \times \B_c} \\\
\O\sn{1} &= \tilde{\P}\sn{1}\V\sn{1} = \text{diag}(\ell\sn{1})^{-1} e^{\S\sn{1} - \m\sn{1}}\V\sn{1} \in \R^{\B_r \times d} \\
\m\sn{2} &= \max(\m\sn{1}, \text{rowmax}(\S\sn{2})) = \m \\
\ell\sn{2} &= e^{\m\sn{1} - \m\sn{2}} \ell\sn{1} + \text{rowsum}(e^{\S\sn{2}-\m\sn{2}}) = \text{rowsum}(e^{\S\sn{1}-\m}) + \text{rowsum}(e^{\S\sn{2}-\m})=\ell \\
\tilde{\P}\sn{2} &= \text{diag}(\ell\sn{2})^{-1}e^{\S\sn{2}-\m\sn{2}} \\ 
\O\sn{2} &= \text{diag}(\ell\sn{1} / \ell\sn{2})^{-1} \O\sn{1} + \tilde{\P}\sn{2}\V\sn{2} = \text{diag}(\ell\sn{2})^{-1}e^{\S\sn{1}-\m}\V\sn{1} + \text{diag}(\ell\sn{2})^{-1}e^{\S\sn{2}-\m}\V\sn{2} = \O
\end{align*}
$$

## FlashAttention-2

A100 GPU에서 FP16 연산 시 Tensor Core는 최대 312 TFLOPs/s의 연산 성능을 제공하는 반면, FP32 연산을 수행하는 CUDA Core는 최대 19.5 TFLOPs/s에 그칩니다.

 따라서, 행렬곱 연산은 가능한 한 Tensor Core에서 수행하도록 하고, softmax와 같은 CUDA Core 기반 연산은 계산량을 최소화하는 것이 바람직합니다. 이러한 원칙에 따라, FlashAttention-1의 온라인 softmax 알고리즘은 이러한 측면에서 개선되었습니다.

출력을 업데이트 할때 두 항을 $\text{diag}(\ell^{(2)})^{-1}$로 조정할 필요가 없습니다. 

$$
\newcommand{\sn}[1]{^{(#1)}}
\newcommand{\m}{\boldsymbol{m}} 
\newcommand{\S}{\mathbf{S}} 
\newcommand{\V}{\mathbf{V}} 
\newcommand{\O}{\mathbf{O}} 

\O\sn{2} = \text{diag}(\ell\sn{1} / \ell\sn{2})^{-1}\O\sn{1} + \text{diag}(\ell\sn{2})^{-1}e^{\S\sn{2}-\m\sn{2}}\V\sn{2}
$$

그 대신 $\mathbf{O}^{(1)}$ 의 "un-scaled" 버전을 유지하고 $\ell^{(2)}$와 같은 통계 정보를 보관할 수 있습니다. 

$$
\newcommand{\sn}[1]{^{(#1)}}
\newcommand{\m}{\boldsymbol{m}} 
\newcommand{\S}{\mathbf{S}} 
\newcommand{\V}{\mathbf{V}} 
\newcommand{\O}{\mathbf{O}} 
\tilde{\O}\sn{2} = \text{diag}(\ell\sn{1})^{-1}\O\sn{1} + e^{\S\sn{2} - \m\sn{2}}\V\sn{2}.
$$

이는 스케일링 된 출력 값을 직접 저장하지 않고, 루프의 마지막에서 $\tilde{\mathbf{O}}^{(\text{last})}$ 를 $\text{diag}(\ell^{(\text{last})})^{-1}$로 스케일링하여 올바른 출력을 얻습니다. 

$$
\newcommand{\sn}[1]{^{#1}}
\newcommand{\m}{\boldsymbol{m}} 
\newcommand{\B}{\boldsymbol{B}} 
\newcommand{\P}{\mathbf{P}} 
\newcommand{\S}{\mathbf{S}} 
\newcommand{\Q}{\mathbf{Q}} 
\newcommand{\K}{\mathbf{K}} 
\newcommand{\V}{\mathbf{V}} 
\newcommand{\R}{\mathbb{R}}
\newcommand{\O}{\mathbf{O}} 
\newcommand{\rowmax}{\text{rowmax}} 
\newcommand{\rowsum}{\text{rowsum}} 
\newcommand{\diag}{\text{diag}} 

\begin{align*}
\m\sn{(1)} &= \rowmax(\S\sn{(1)}) \in \R\sn{\B_r} \\
\ell\sn{(1)} &= \rowsum(e^{\sn{(1)}-\m\sn{(1)}}) \in \R\sn{\B_r} \\
\tilde{\O}\sn{(1)} &= e\sn{\S\sn{(1)}-\m\sn{(1)}}\V\sn{(1)} \in \R\sn{\B_r \times d} \\
\m\sn{(2)} &= \max(\m\sn{(1)}, \rowmax{\S\sn{(2)}}) =\m \\
\ell\sn{(2)} &= e\sn{\m\sn{(1)}-\m\sn{(2)}}\ell\sn{(1)} + \rowsum(e\sn{\S\sn{(2)}-\m\sn{(2)}}) = \rowsum(e\sn{\S\sn{(1)}-\m}) + \rowsum(e\sn{\S\sn{2}-m}) = \ell\\
\tilde{\P}\sn{(2)} &= \diag(\ell\sn{(2)})\sn{-1}e\sn{\S\sn{(2)}-\m\sn{(2)}} \\
\tilde{\O}\sn{(2)} &= \diag(e\sn{\m\sn{(1)}-\m\sn{(2)}})\sn{-1}\tilde{\O}\sn{(1)} + e\sn{\S\sn{(2)}-\m\sn{(2)}}\V\sn{(2)} = e\sn{\S\sn{(1)}-\m}\V\sn{(1)} + e\sn{\S\sn{(2)}-\m}\V\sn{(2)} \\
\O\sn{(2)} &= \diag(\ell\sn{(2)})\sn{-1}\tilde{\O}\sn{(2)} = \O.

\end{align*}
$$

### Algorithm 1 FlashAttention-2 forward pass 

$$
\newcommand{\sp}[1]{^{#1}}
\newcommand{\spp}[1]{^{(#1)}}
\newcommand{\m}{\boldsymbol{m}} 
\newcommand{\B}{\boldsymbol{B}} 
\newcommand{\P}{\mathbf{P}} 
\newcommand{\S}{\mathbf{S}} 
\newcommand{\Q}{\mathbf{Q}} 
\newcommand{\K}{\mathbf{K}} 
\newcommand{\V}{\mathbf{V}} 
\newcommand{\R}{\mathbb{R}}
\newcommand{\O}{\mathbf{O}} 
\newcommand{\rowmax}{\text{rowmax}} 
\newcommand{\rowsum}{\text{rowsum}} 
\newcommand{\diag}{\text{diag}}
\newcommand{\quadd}{\quad\quad}
\newcommand{\quaddd}{\quad\quad\quad}
\newcommand{\qua}{\hspace{0.66em}}

\begin{flalign*}
& \text{Require: Matrices } \Q, \K, \V \in \R\sp{N \times d} \text{ in HBM, block sizes } \B_c, \B_r. \\
& 1.\text{ Divide } \Q \text{ into } T_r = \lceil N / B_r \rceil \text{ blocks } \Q_1, \dots, \Q_T, \text{ each of size } \B_r \times d, \\
& \quad \text{and divide } \K, \V \text{ into } T_c = \lceil N / B_c \rceil \text{ blocks } \K_1, \dots, \K_{T_c} \text{ and } \V_1, \dots, \V_{T_c}, \\
& \quad \text{each of size } \B_c \times d. \\

& 2. \text{ Divide the output }\O \in \R\sp{N \times d} \text{ into } T_r \text{ blocks } \O_i, \dots, \O_{T_r} \text{ of size } \B_r \times d \text{ each,} \\
& \quad \text{and divide the log-sum-exp } L \text{ into } T_r \text{ blocks } L_i \dots, L_{T_r} \text{ of size } \B_r \text{ each}. \\

& 3. \text{ for } 1 \leq i \leq T_r \text{ do} \\
& 4. \quad \text{ Load } \Q_i \text{ from HBM to on-chip SRAM.} \\
& 5. \quad \text{ On chip, initialize } \O_i\spp{0} = (0)_{\B_r \times d} \in \R\spp{\B_r \times d}, \ell_i\spp{0} = (0)_{\B_r} \in \R\sp{\B_r}, \m_i\spp{0} = (-\infty)_{\B_r} \in \R\sp{\B_r} \\
& 6. \quad \text{ for } 1 \leq j \leq T_c \text{ do } \\
& 7. \quadd\text{ Load } \K_j, \V_j \text{ from HBM to on-chip SRAM } \\
& 8. \quadd \text{ On chip, compute } \S_i\spp{j} = \Q_i\K_j^T \in \R\sp{\B_r \times \B_c }. \\
& 9. \quadd \text{ On chip, compute } \m_i\spp{j} = \max(\m_i\spp{j-1}, \rowmax(\S_i\spp{j})) \in \R\sp{\B_r}, \\
& \quaddd \tilde{\P}_i\spp{j} = \exp(\S_i\spp{j}-\m_i\spp{j}) \in \R\sp{\B_r \times \B_c} \\
& \quaddd \text{(pointwise) } \ell_i\spp{j} = e\sp{\m_i\sp{j-1}-m_i\spp{j}} \ell\spp{j-1} + \rowsum(\tilde{\P}_i\spp{j}) \in \R\sp{\B_r} \\
& 10. \quad \qua \text{On chip, compute } \O_i =  \diag(e\sp{\m_i\spp{j-1}-\m_i\spp{j}})\sp{-1}\O_i\spp{j-1}+\tilde{P}_i\spp{j}\V_j \\
& 11. \quad \qua \text{end for} \\
& 12. \qua \text{ On chip, compute } \O_i\spp{j} = \diag(\ell_i\spp{T_c})\sp{-1}\O_i\spp{T_c} \\
& 13. \qua \text{ On chip, compute } L_i = \m_i\spp{T_c} + \log(\ell_i\spp{T_c}) \\
& 14. \qua \text{ Write } \O_i \text{ to HBM as the } i \text{-th block of } \O \\
& 15. \qua \text{ Write } L_i \text{ to HBM as the } i \text{-th block of } L \\
& 16. \text{ end for } \\
& 17. \text{ Return the Output } \O \text{ and the log-sum-exp } L.


\end{flalign*}
$$

