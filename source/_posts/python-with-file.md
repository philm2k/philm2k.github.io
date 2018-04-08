---
title: 파일 다루기(Treat with files, folders, etc)
categories:
  - Python
tags:
  - H2
  - files
  - folders
date: 2018-03-31 17:43:02
thumbnail:
---

# 특정 폴더에 가장 최근에 생성된 파일명 가져오기

## 1. glob.glob을 이용하여 파일목록 가져오기
{% codeblock lang:python %}
from glob import glob
lst_files = glob("D:/Dev/02. Python Data Analysis/DataScienceHandbook/*.ipynb")
lst_files[:5]
{% endcodeblock %}
다음 그림은 그 결과
{% asset_img glob.glob.jpg 특정폴더의 파일목록 가져오기 %}

## 2. getctime을 이용하여 가장 최근에 생성된 파일명 찾기
{% codeblock lang:python %}
from os.path import getctime
latest_file = max(lst_files, key=getctime)
latest_file
{% endcodeblock %}
다음 그림은 그 결과
{% asset_img getctime.jpg getctime을 이용하여 가장 최근에 생성된 파일명 찾기 %}


### Related Posts