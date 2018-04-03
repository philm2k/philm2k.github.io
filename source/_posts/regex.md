---
title: Python 정규표현식 기본
categories:
  - Python
tags:
  - Regular Expressions
date: 2018-02-25 23:49:54
thumbnail:
---
# 정규표현식을 이용하여 요즘 하고 있는 일

{% codeblock lang:python %}
import re
icRegex = re.compile(r'BOCOM\d\d\d\d\d')
txt = "This is 책임자승인(BOCOM00001)!!! That is 책임자승인(BOCOM00201)!!!"
mo = icRegex.search(txt)
print(mo.group())
{% endcodeblock %}
이 코드를 실행하면 결과는 "BOCOM00001"입니다.

txt의 문장과 같은 코드가 들어 있는 약 20,000개의 파일을 하나씩 열어서 정규표현식으로 찾고 pandas를 이용하여 Excel로 저장하는데 10분이 채 안 걸립니다. ^^

위의 내용을 잘 보시면 다음의 단계를 거친 것을 알 수 있습니다.

## Python의 정규표현식 사용의 단계

- import re로 정규식 모듈을 가져온다.
- re.compile() 함수로 Regex 개체를 만든다.
- 검색할 문자열을 Regex 개체의 search() 메소드로 전달하여 Match 객체를 돌려받는다.
- Match 개체의 group()메소드를 호출해서 실제 일치하는 텍스트 문자열을 돌려받는다.

## search() vs. findall()

- search()는 검색하는 문자열에서 처음으로 나타나는 일치하는 텍스트의 Match 개체를 return
- findall() 메소드는 검색 문자열에서 일치하는 모든 문자열을 return

다시 말해서 위의 코드에서 search() 대신 findall()을 사용할 경우의 답은 ['BOCOM00001', 'BOCOM00201']입니다.
코드는 약간 다릅니다. 맞게 써보면 다음과 같습니다.
{% codeblock lang:python %}
import re
icRegex = re.compile(r'BOCOM\d\d\d\d\d')
txt = "This is 책임자승인(BOCOM00001)!!! That is 책임자승인(BOCOM00201)!!!"
print(icRegex.findall(txt))
{% endcodeblock %}

### Related Posts

{% blockquote Dave Child https://www.cheatography.com/davechild/cheat-sheets/regular-expressions/ Regular Expression Cheatsheet %}Dave Child의 정규표현식에 대한 요약 자료입니다. {% endblockquote %}