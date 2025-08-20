---
layout: single
title: "FlashAttention-2의 수식 오류 정리"
categories: "코딩"
toc: true
typora-root-url: ./typora-root-url
---



## FlashAttention-2의 수식 오류 정리

Flashattention-2 논문의 수식 오류를 다룹니다.

## 3.1.1 Forward pass

- FlashAttetnion-2는 un-scaled  version의  $\mathbf{O}^{(2)}$ 에 대해 다음과 같이 언급하고 있다. 

$$
\tilde{\mathbf{O}}^{(2)} = \text{diag}(\mathcal{l}^{(1)})^{-1}\mathbf{O}^{(1)} + \text{e}^{\mathbf{S}^{(2)}-m^{{(2)}}}\mathbf{V}^{(2)}
$$

- 해당 식에서 $\tilde{\mathbf{P}}^{(2)}$ 가 언급되지 않는 오류 및 $\mathcal{l}^{(1)}$ 및 $\tilde{\mathbf{O}}^{(1)}$ 에 대한 오류가 발견되었다. 실제 수식은 다음과 같다. 

$$
\tilde{\mathbf{O}}^{(2)} = \text{e}^{\mathbf{S}^{(2)}-m^{{(2)}}}\tilde{\mathbf{O}}^{(1)} + \tilde{\mathbf{P}}^{(2)}\mathbf{V}^{(2)}
$$

### Reference 

- https://github.com/Dao-AILab/flash-attention/issues/991