---
title: generator
categories:
  - Python
tags:
  - generator
  - yield
  - yield from
date: 2018-03-12 00:26:34
thumbnail:
---
# Python Cookbook 4.14

## 방대한 양의 로그 파일이 들어 있는 디렉터리에서 작업

{% codeblock lang:python %}
import os
import fnmatch
import gzip
import bz2
import re

def gen_find(filepat, top):
    '''
    디렉터리 트리에서 와일드카드 패턴에 매칭하는 모든 파일 이름을 찾는다.
    '''
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            yield os.path.join(path, name)

def gen_opener(filenames):
    '''
    파일 이름 시퀀스를 하나씩 열어 파일 객체를 생성한다.
    다음 순환으로 넘어가는 파일을 닫는다.
    '''
    for filename in filenames:
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rt')
        elif filename.endswith('.bz2'):
            f = bz2.open(filename, 'rt')
        else:
            f = open(filename, 'rt')
        yield f
        f.close()

def gen_concatenate(iterators):
    '''
    이터레이터 시퀀스를 합쳐 하나의 시퀀스를 만든다.
    '''
    for it in iterators:
        yield from it

def gen_grep(pattern, lines):
    '''
    라인 시퀀스에서 정규식 패턴을 살펴본다.
    '''
    pat = re.compile(pattern)
    for line in lines:
        if pat.search(line):
            yield line
            
{% endcodeblock %}

## 로그 분석(?)

이제 이 함수들을 이용해 python이란 단어를 포함하고 있는 모든 로그 라인을 찾으려면 다음과 같이 한다.

{% codeblock lang:python %}
lognames = gen_find('access-log*', 'www')
files = gen_opener(lognames)
lines = gen_concatenate(files)
pylines = gen_grep('(?i)python', lines)
for line in pylines:
    print(line)
{% endcodeblock %}

또는 제너레이터 표현식을 써서 다음과 같이 확장할 수 있다.
{% codeblock lang:python %}
lognames = gen_find('access-log*', 'www')
files = gen_opener(lognames)
lines = gen_concatenate(files)
pylines = gen_grep('(?i)python', lines)
bytecolumn = (line.rsplit(None, 1)[1] for line in pylines)
bytes = (int(x) for x in bytecolumn if x != '-')
print('Total', sum(bytes))
{% endcodeblock %}

## Related Posts