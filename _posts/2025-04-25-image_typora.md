---
layout: single
title: "Typora 이미지 경로 문제"
categories: "일상"
toc: true
typora-root-url: ./typora-root-url
---
현재 이 블로그는 [Typora](https://typora.io/)를 사용해 작성하고 있습니다. 

이번 글에서는 Typora와 깃허브 호스팅 양쪽 모두에서 이미지가 렌더링하기 위한 시행착오를 정리해보았습니다.

## 이미지 경로 접근 방식

이미지 삽입에는 일반적으로 다음 두 가지 방식이 있습니다.

- 상대 경로 
  - 현재 `.md` 파일의 위치를 기준으로 이미지 경로를 지정 
  - 예: `../../images`

- 절대 경로 
  - 프로젝트 루트를 기준으로 경로를 지정 
  - 예: `/images`

## 문제 발생 

이미지를 삽입하면서 다음과 같은 문제가 발생했습니다
- 상대 경로 
  - 카테고리 기능으로 인해 타이포라는 `../images` , 깃허브 호스팅은 `../../images` 로 접근 경로가 달라집니다. 

- 절대 경로 
  - 타이포라는 `/images`, 깃허브 호스팅은 `https://osehyeon.github.io/blog/images`로 접근 경로가 달라집니다. 


## 시도한 접근 

- `{{ site.baseurl }}/images`를 사용하는 방식  
  - 깃허브 호스팅에서는 정상 작동되나 Typora에서는 렌더링되지 않았습니다.

## 최종 해결 방법

- `typora-root-url: ./typora-root-url`을 설정해, Typora가 사용하는 이미지 기준 경로를 가상으로 지정하였습니다.
- 모든 이미지 경로를 `../../images` 형태로 통일하였습니다.
  - Typora에서는 `typora-root-url` 기준으로 이미지가 정상 표시됩니다. 
  - GitHub Pages에서는 포스트 경로 구조상 `../../images`가 실제 경로와 일치해 문제 없이 렌더링됩니다. 

