---
title: datetime
categories:
  - Python
tags:
  - datetime
date: 2018-03-01 21:59:26
thumbnail:
---

# 다양한 형태의 날짜 정보 가져오기
아래 세가지 형태의 날짜 형태만 가져오면 문제가 없더군요.. ^^
{% codeblock lang:python %}
from datatime import datetime
dt0 = datetime.now().strftime('%y%m%d')       # 180301
dt1 = datetime.now().strftime('%Y%m%d')       # 20180301
dt2 = datetime.now().strftime('%Y%m%d-%H%M')  # 20180301-22:04
{% endcodeblock %}

### Related Posts