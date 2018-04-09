---
title: python-how-to(S)
categories:
  - Python
  - 용어
tags:
  - H2
  - S
date: 2018-04-09 13:54:35
thumbnail:
---
# S

## selenium

## sequence 자료형

여러 원자로(숫자형은 하나의 원자로 이루어져 있다.) 구성된 자료형을 시퀀스(Sequence) 자료형이라고 부른다.

### 1. sequence 종류

- 컨테이너 시퀀스(container sequence): 서로 다른 자료형의 항목들을 담을 수 있음
  - 리스트: Mutable
  - 튜플: Immutable
  - collections.deque: Mutable

- 균일 시퀀스(flat sequence): 단 하나의 자료형만 담을 수 있음
  - str: Immutable
  - bytes: Immutable
  - bytearray: Mutable
  - memoryview: Mutable
  - array.array: Mutable

### 2. sequence의 특징

- 동일한 타입의 원소를 가질 수도 있고 리스트처럼 객체를 원소로 가질 수도 있다.
- Mutable과 Immutable이 있다.
- 순서가 있으므로 인섹스를 이용하여 검색이 가능하고 슬라이스로 부분도 검색이 가능하다.

## set

- Set 자료형
- 원소들을 검색할 때 index가 아닌 key로 접근해서 처리
- 키는 유일해야 함
- 따라서 key를 생성할 때 hash 알고리즘을 통해 유일한 값만 구성

### Related Posts