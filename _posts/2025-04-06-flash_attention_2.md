---
layout: single
title:  "[논문 리뷰] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
categories: "논문"
toc: true
typora-root-url: .
---

"FlashAttention-2" 논문은 프린스턴 대학교 박사 과정 학생 Tri Dao가 제안한 연구로, 트랜스포머 모델에서 긴 시퀀스를 처리할 때 셀프 어텐션 메커니즘이 가지는 메모리 및 계산 비효율성 문제를 극복하고자 개발되었습니다. 이 논문은 2023년 arXiv에 처음 공개되었으며, 이후 ICLR 2024에 포스터 형식으로 채택되어 발표되었습니다. 

## Standard Attetnion Implementation
