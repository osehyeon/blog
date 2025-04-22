---
layout: single
title:  "FlashAttenion의 Logsumexp 분석"
categories: "코딩"
toc: true
typora-root-url: .
---

[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)의 [Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py) 예제의 `_attn_fwd` 를 확인할 경우 다음과 같은 코드를 확인할 수 있다. 

```python
m_i += tl.math.log2(l_i)
acc = acc / l_i[:, None]
m_ptrs = M + off_hz * N_CTX + offs_m
tl.store(m_ptrs, m_i)
tl.store(O_block_ptr, acc.to(Out.type.element_ty))
```

여기서 최종적으로  `acc`와 `m_i` 를 글로벌 메모리에 저장을 한다. 

`acc`는 어텐션$ \left(\text{softmax}(\frac{QK^T}{\sqrt(d_k)})V \right )$의 결과임으로 이를 저장하는 것은 직관적이나  `m_i`를 저장하는 것은 직관적이지 않다. 

이를 위해 `Logsumexp` 과 Softmax의 Backward를 알아야 한다. 

## Logsumexp and Softmax Backward

softmax는 $\mathbf{x} = (x_1, x_2, \dots, x_n)$ 에 대해서 다음과 같이 정의된다.

$$
p_i = \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^n \exp(x_j)} = \frac{\exp(x_i)}{Z}
$$

`Logsumexp`는 다음과 같은 형태의 수학 함수를 가르킨다.

$$
\text{logsumexp}(x_1, x_2, \dots, x_n) \coloneqq \log \left (\sum_{j=1}^n \exp(x_j) \right )
$$

위와 같은 형태는 $x$ 값의 크기에 따라 **overflow** 및 **underflow** 문제가 발생한다. 수치를 **안정화**하기 위해 재정렬이 필요하다. 

$$
\text{Let } m_i = \max_j x_j
$$

$$
\log \left( \sum_j \exp(x_j) \right) = \log \left( \sum_j \exp(x_j - m_i + m_i) \right) = \log\left( \exp(m_i) \cdot \sum_j \exp(x_j - m_i) \right)
$$

$$
\log\left( \exp(m_i) \cdot \sum_j \exp(x_j - m_i) \right) = \log(\exp(m_i)) + \log\left( \sum_j \exp(x_j-m_i) \right) = m_i + \log\left( \sum_j \exp(x_j - m_i) \right)
$$

softmax는 `logsumexp`로 표현할 수 있다. 

$$
p_i = \frac{\exp(x_i)}{Z} = \exp(x_i - \log Z)$\ \log Z = \text{logsumexp}(x_1, x_2, \dots, x_n)$  임으로 최종적으로 다음과 같이 정의된다.
$$

$$
p_i = \exp(x_i - \text{logsumexp}(x_1, x_2, \dots, x_n))
$$

이때의 `logsumexp`가 위 코드에서 저장하는 `m_i += tl.math.log2(l_i)`와 같다. 

Softmax의 Backward의 식은 다음과 같다. 

$$
\frac{\partial L}{\partial x_i} = p_i \left( \frac{\partial L}{\partial p_i} - \sum_j p_j \frac{\partial L }{\partial p_j} \right)
$$

이를 계산하기 위해서는  $p_i$ 의 값이 필요하다. 이를 forward 계산 때 $p_i$를 저장하는 건, $\text{CTX} \times \text{CTX}$ 만큼의 용량이 필요하다. 만약 `logsumexp`를 저장하여, 이를 통해 Backward에서 $p_i$를 계산한다면, $\text{CTX}$  만큼의 용량만 있으면 된다. 
