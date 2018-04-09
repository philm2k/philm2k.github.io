---
title: Python bool에 관한 정리
categories:
  - Python
tags:
  - bool
date: 2018-04-09 12:36:21
thumbnail:
---

# 파이선 bool이 False인 경우들

None, 숫자0, 빈문자열, 빈리스트, 빈 튜플, 빈 사전, set() 

```python
for i in [0, 0.0, "", (),{},[], set(), None]:
    print(str(i), bool(i))
```

    0 False
    0.0 False
     False
    () False
    {} False
    [] False
    set() False
    None False
    

### Related Posts