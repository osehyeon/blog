---
layout: single
title: "트랜스포머 기반 모델의 구조"
categories: "코딩"
toc: true
typora-root-url: ./typora-root-url
---

트랜스포머 기반 모델의 구조가 어떻게 발전해왔는지에 대해 다룬다. 

## Attention is All You Need 

2017년 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 논문을 통해 트랜스포머 구조가 처음 제안되어 널리 알려지게 되었다.

초기에 제안된 Attention과 FFN 조합은 대부분의 트랜스포머 변형 모델에서 유지되었으며, 이후 연구는 이를 보다 효과적으로 구현하기 위한 최적화에 초점을 맞추었다.

### The Transformer - model architecture

트랜스포머는 입력을 해석하는 인코더 블록과 출력을 생성하는 디코더 블록으로 구성된다.

<p align="center">
  <img src="../../images/2025-04-30-attention/image-20250501011034649.png" width="50%">
</p>


### Attention

스케일드 닷 프로덕트 어텐션은 단어 간의 관련도를 점수로 계산해, 중요한 정보에 더 집중하게 한다.

멀티헤드 어텐션은 여러 어텐션을 동시에 사용해, 문장을 다양한 관점에서 이해할 수 있게 한다.

<p align="center">
  <img src="../../images/2025-04-30-attention/image-20250501012658298.png" width="80%">
</p>


### FFN

FFN은 각 단어의 표현에 비선형 함수를 적용해, 표현력을 확장하는 역할을 한다.

$$
\text{FFN}(x) = \max(0, xW_1 +b_1)W_2+b_2
$$

## Improving Language Understanding by Generative Pre-Training

2018년 [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) 논문을 통해 알려졌다. 

GPT-1은 Attention is All You Need 논문의 디코더 구조를 기반으로 하지만 Cross-Attention을 제거하고 causal self-attention만 사용한다. 

<p align="center">
  <img src="../../images/2025-04-30-attention/image-20250501013607925.png" width="100%">
</p>


## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

2018년 [BERT](https://arxiv.org/abs/1810.04805) 논문을 통해 알려졌다. 

BERT는 Attention is All You Need 논문의 인코더 구조를 기반으로 한다. 

<p align="center">
  <img src="../../images/2025-04-30-attention/bert.png" width="25%">
</p>


FFN의 비선형 함수로 ReLU 대신에 [GeLU](https://www.semanticscholar.org/paper/4361e64f2d12d63476fdc88faf72a0f70d9a2ffb)를 사용하고, 트랜스포머 블록 이후에 Norm이 한 번 더 추가되었다. 

$$
\text{FFN}(x) = \text{GELU}(0, xW_1 +b_1)W_2+b_2
$$

## Language Models are Unsupervised Multitask Learners

2019년 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 논문을 통해 알려졌다. 

GPT-2는 트랜스포머 블록에 Post-Norm 대신 [Pre-Norm](https://arxiv.org/pdf/1603.05027) 을 적용하였고, 트랜스포머 블록 이후에 Norm이 한 번 더 추가되었다.

<p align="center">
  <img src="../../images/2025-04-30-attention/image-20250501030454350.png" width="20%">
</p> 

## Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
2019년 [T5](https://arxiv.org/abs/1910.10683) 논문을 통해 알려졌다. 

Attention is All You Need 논문의 인코더-디코더 구조를 기반으로 한다. 

GPT-2와 같이 트랜스포머 블록에 Pre-Norm을 적용하였고, FFN의 비선형 함수로 ReLU를 사용하였다. 

Absolute Positional Embedding이 아닌 [Relative Position Encoding](https://arxiv.org/abs/1803.02155)를 Relative Position Bias로 단순화하여 attention score에 적용하였다. 

$$
e_{ij} = \frac{(x_i W^Q)(x_j W^K)^T}{\sqrt{d_k}} + b_{i-j}
$$
