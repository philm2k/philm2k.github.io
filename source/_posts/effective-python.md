---
title: effective python 파이썬 코딩의 기술[책]
categories:
  - python
  - books
tags:
  - effective
date: 2018-04-14 21:56:37
thumbnail:
---

이 책을 보는 중입니다. 생각할 것이 많더군요
다음은 목차입니다.

# 1장 ▶ 파이썬다운 생각 015 

- Better way 1 사용 중인 파이썬의 버전을 알자 016 
- Better way 2 PEP 8 스타일 가이드를 따르자 017 
- Better way 3 bytes, str, unicode의 차이점을 알자 020 
- Better way 4 복잡한 표현식 대신 헬퍼 함수를 작성하자 024 
- Better way 5 시퀀스를 슬라이스하는 방법을 알자 028 
- Better way 6 한 슬라이스에 start, end, stride를 함께 쓰지 말자 032 
- Better way 7 map과 filter 대신 리스트 컴프리헨션을 사용하자 034 
- Better way 8 리스트 컴프리헨션에서 표현식을 두 개 넘게 쓰지 말자 036 
- Better way 9 컴프리헨션이 클 때는 제너레이터 표현식을 고려하자 039 
- Better way 10 range보다는 enumerate를 사용하자 041 
- Better way 11 이터레이터를 병렬로 처리하려면 zip을 사용하자 043 
- Better way 12 for와 while 루프 뒤에는 else 블록을 쓰지 말자 046 
- Better way 13 try/except/else/finally에서 각 블록의 장점을 이용하자 049 
  - finally 블록 050 
  - else 블록 050 
  - 모두 함께 사용하기 051 

# 2장 ▶ 함수 053 

- Better way 14 None을 반환하기보다는 예외를 일으키자 054 
- Better way 15 클로저가 변수 스코프와 상호 작용하는 방법을 알자 057 
  - 데이터 얻어오기 060 
  - 파이썬 2의 스코프 062 
- Better way 16 리스트를 반환하는 대신 제너레이터를 고려하자 063 
- Better way 17 인수를 순회할 때는 방어적으로 하자 066 
- Better way 18 가변 위치 인수로 깔끔하게 보이게 하자 072 
- Better way 19 키워드 인수로 선택적인 동작을 제공하자 075 
- Better way 20 동적 기본 인수를 지정하려면 None과 docstring을 사용하자 079 
- Better way 21 키워드 전용 인수로 명료성을 강요하자 083 
  - 파이썬 2의 키워드 전용 인수 086 

# 3장 ▶ 클래스와 상속 089 

- Better way 22 딕셔너리와 튜플보다는 헬퍼 클래스로 관리하자 090 
  - 클래스 리팩토링 093 
- Better way 23 인터페이스가 간단하면 클래스 대신 함수를 받자 097 
- Better way 24 객체를 범용으로 생성하려면 @classmethod 다형성을 이용하자 102 
- Better way 25 super로 부모 클래스를 초기화하자 108 
- Better way 26 믹스인 유틸리티 클래스에만 다중 상속을 사용하자 114 
- Better way 27 공개 속성보다는 비공개 속성을 사용하자 119 
- Better way 28 커스텀 컨테이너 타입은 collections.abc의 클래스를 상속받게 만들자 126 

# 4장 ▶ 메타클래스와 속성 133 

- Better way 29 게터와 세터 메서드 대신에 일반 속성을 사용하자 134 
- Better way 30 속성을 리팩토링하는 대신 @property를 고려하자 139 
- Better way 31 재사용 가능한 @property 메서드에는 디스크립터를 사용하자 144 
- Better way 32 지연 속성에는 _ _getattr_ _, _ _getattribute_ _, _ _setattr_ _을 사용하자 151 
- Better way 33 메타클래스로 서브클래스를 검증하자 158 
- Better way 34 메타클래스로 클래스의 존재를 등록하자 161 
- Better way 35 메타클래스로 클래스 속성에 주석을 달자 167 

# 5장 ▶ 병행성과 병렬성 171 

- Better way 36 자식 프로세스를 관리하려면 subprocess를 사용하자 172 
- Better way 37 스레드를 블로킹 I/O용으로 사용하고 병렬화용으로는 사용하지 말자 178 
- Better way 38 스레드에서 데이터 경쟁을 막으려면 Lock을 사용하자 183 
- Better way 39 스레드 간의 작업을 조율하려면 Queue를 사용하자 188 
  - Queue로 문제 해결하기 192 
- Better way 40 많은 함수를 동시에 실행하려면 코루틴을 고려하자 197
  - 생명 게임 200 
  - 파이썬 2의 코루틴 207 
- Better way 41 진정한 병렬성을 실현하려면 concurrent.futures를 고려하자 209 

# 6장 ▶ 내장 모듈 215 

- Better way 42 functools.wraps로 함수 데코레이터를 정의하자 216 
- Better way 43 재사용 가능한 try/finally 동작을 만들려면 contextlib와 with 문을 고려하자 219 
  - with 타깃 사용하기 221 
- Better way 44 copyreg로 pickle을 신뢰할 수 있게 만들자 223 
  - 기본 속성 값 226 
  - 클래스 버전 관리 228 
  - 안정적인 임포트 경로 229 
- Better way 45 지역 시간은 time이 아닌 datetime으로 표현하자 231 
  - time 모듈 232 
  - datetime 모듈 234
- Better way 46 내장 알고리즘과 자료 구조를 사용하자 237 
  - 더블 엔디드 큐 237 
  - 정렬된 딕셔너리 238 
  - 기본 딕셔너리 239 
  - 힙 큐 240 
  - 바이섹션 241 
  - 이터레이터 도구 242 
- Better way 47 정밀도가 중요할 때는 decimal을 사용하자 243 
- Better way 48 커뮤니티에서 만든 모듈을 어디서 찾아야 하는지 알아두자 247 

# 7장 ▶ 협력 249 

- Better way 49 모든 함수, 클래스, 모듈에 docstring을 작성하자 250 
  - 모듈 문서화 251 
  - 클래스 문서화 252 
  - 함수 문서화 253 
- Better way 50 모듈을 구성하고 안정적인 API를 제공하려면 패키지를 사용하자 255 
  - 네임스페이스 256 
  - 안정적인 API 258 
- Better way 51 루트 Exception을 정의해서 API로부터 호출자를 보호하자 262 
- Better way 52 순환 의존성을 없애는 방법을 알자 266 
  - 임포트 재정렬 268 
  - 임포트, 설정, 실행 269 
  - 동적 임포트 271 
- Better way 53 의존성을 분리하고 재현하려면 가상 환경을 사용하자 273 
  - pyvenv 명령 275 
  - 의존성 재현 277 

# 8장 ▶ 제품화 281 

- Better way 54 배포 환경을 구성하는 데는 모듈 스코프 코드를 고려하자 282 
- Better way 55 디버깅 출력용으로는 repr 문자열을 사용하자 285 
- Better way 56 unittest로 모든 것을 테스트하자 289 
- Better way 57 pdb를 이용한 대화식 디버깅을 고려하자 293 
- Better way 58 최적화하기 전에 프로파일하자 295 
- Better way 59 tracemalloc으로 메모리 사용 현황과 누수를 파악하자 301 

# 찾아보기 305

### Related Posts