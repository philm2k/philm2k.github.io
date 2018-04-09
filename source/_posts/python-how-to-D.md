---
title: Python How-To(D)
categories:
  - Python
  - 용어
tags:
  - H2
  - D
date: 2018-03-22 00:00:58
thumbnail:
---
# D

## Data Model

- 숫자형
- 시퀀스형
- 매핑형
  - 딕셔너리(dict): Mutable
- Set형
  - set: Mutable
  - frozenset: Immutable

## decorator

함수를 인자로 받아서 추가적인 작업을 함수 전/후에 해서 인자로 받은 함수를 포함한 결과를 return하는 함수
{% codeblock lang:python %}
from functools import wraps
def decorater_function(func):
    @wraps
    def inside_deco_function():
        print("Do something befor func()")
        func()
        print("Do something after func()")
    return inside_deco_function
{% endcodeblock %}

목적은 이미 만들어져 있는 기존의 함수를 수정하지 않고도 래퍼(Wrapper)함수를 이용하여 여러가지 기능을 추가할 수 있기 때문입니다. 예를 들어

- 인증처리
- 로그 남기는 기능 추가 등

사용방법은 다음과 같습니다.
{% codeblock lang:python %}
@decorator_function
def add_func(x):
    return x + x
{% endcodeblock %}

## dict

- 매핑 자료형
- 원소들을 검색할 때 index가 아닌 key로 접근해서 처리
- 키는 유일해야 함
- 따라서 key를 생성할 때 hash 알고리즘을 통해 유일한 값만 구성

## dict.get(...) vs. dict.setdefault

### dict.get(key, defaultvalue)

첫 번째 인자를 가지고 dict에서 value를 가지고 오는 함수
두 번째 인자가 없다면 key에 해당하는 value가 없을 경우 None을 반환, 두 번째 인자를 지정하는 경우 key에 해당하는 value가 없는 경우 defaultvalue를 가져 옴
{% codeblock lang:python %}
>>> aaa = {'name':'hong', 'phone':'01012345678', 'birth':'0323'}
>>> a.get('name')
'hong'
>>> a.get('phone')
'01012345678'
{% endcodeblock %}

### dict.setefault

검색키가 존재하면 해당 키에 대한 값을 가져 오고, 존재하지 않으면 기본값으로 해당 키를 생성한 후 기본값을 반환

## Duck Typing

{% blockquote 루시아누 하말류, 전문가를 위한 파이썬(fluent Python) 부록B 파이선 용어 %}
객체의 클래스나 인터페이스 선언에 상관없이 매서드를 적절히 구현하면 어떠한 객체에도 함수를 호출할 수 있는 다형성(polymorphism)의 한 형태
{% endblockquote %}

{% blockquote 위키백과 https://ko.wikipedia.org/wiki/%EB%8D%95_%ED%83%80%EC%9D%B4%ED%95%91 Duck Typing %}
컴퓨터 프로그래밍 분야에서 덕 타이핑(duck typing)은 동적 타이핑의 한 종류로, 객체의 변수 및 메소드의 집합이 객체의 타입을 결정하는 것을 말한다. 클래스 상속이나 인터페이스 구현으로 타입을 구분하는 대신, 덕 타이핑은 객체가 어떤 타입에 걸맞은 변수와 메소드를 지니면 객체를 해당 타입에 속하는 것으로 간주한다. “덕 타이핑”이라는 용어는 다음과 같이 표현될 수 있는 덕 테스트에서 유래했다. (덕은 영어로 오리를 의미한다.){% endblockquote %}

> 만약 어떤 새가 오리처럼 걷고, 헤엄치고, 꽥꽥거리는 소리를 낸다면 나는 그 새를 오리라고 부를 것이다.

### Related Posts