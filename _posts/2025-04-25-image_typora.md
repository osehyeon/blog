---
layout: single
title: "Typora 이미지 경로 문제"
categories: "일상"
toc: true
typora-root-url: ./typora-root-url
---

현재 해당 블로그는 [Typora](https://typora.io/)를 통해 작성하고 있습니다. 이번 포스팅은 타이포라와 깃허브 호스팅에서 둘 다 이미지를 랜더링하기 위한 시행착오를 담고 있습니다. 

## 이미지 경로 접근 방식

이미지는 두 가지 방법으로 접근을 할 수 있습니다. 

1. 상대 경로 접근: 현재 `.md` 파일의 위치를 기준으로 이미지 파일까지의 경로를 지정합니다. 
2. 절대 경로 접근: 루트를 기준으로 이미지 파일까지의 경로를 지정합니다. 

## 문제 발생  

이미지에 대한 경로 추가하면서 다음과 같은 문제가 발생하였습니다. 

- 상대 경로로 접근할 경우 카테고리 기능으로 인해 타이포라는 `../images` , 깃허브 호스팅은 `../../images` 로 접근 경로가 달라집니다. 
- 절대 경로로 접근할 경우 현재 블로그는 하위 경로를 기준으로 호스팅하고 있어 타이포라는 `/images`, 깃허브 호스팅은 
- ![image-20250425114607174](https://osehyeon.github.io/blog/images/2025-04-26-typora/image-20250425114607174.png)

<p align="center">
  <img src="https://osehyeon.github.io/blog/images/2025-04-25-fp_vit/image-20250425010450030.png" style="width:70%;">
</p>

<p align="center">
  <img src=/blog/images/2025-04-25-fp_vit/image-20250425010450030.png" style="width:70%;">
</p>

<p align="center">
  <img src=/images/2025-04-25-fp_vit/image-20250425010450030.png" style="width:70%;">
</p>
