---
title: Pandas로 엑셀파일 다루기
categories:
  - Data Science
  - Pandas
tags:
  - Excel
  - Python
date: 2018-02-27 00:53:10
thumbnail:
---
기업체나 학교에서도 거의 모든 사람들이 쓰는 소프트웨어가 Excel입니다. 간단한 통계나 보고서용 표 정리를 위해 엑셀의 계산 기능을 많이 들 사용합니다.  
pandas를 이용하시면 다양하게 Excel의 데이터를 가공하고 다시 Excel로 저장이 가능합니다.

다음은 IMDB에서 제공하는 movie data excel 파일입니다. 여기(https://www.dataquest.io/blog/large_files/movies.xls) 에서 다운로드 하실 수 있습니다.
{% asset_img movies.JPG Example %}

# Excel 읽어서 DataFrame에 넣기

{% codeblock lang:python %}
import pandas as np
BASE_PATH = "d:/dev"
FILE_NAME = "abc.xlsx"

from os.path import join

df = pd.read_excel(join(BASE_PATH, FILE_NAME))
df.head()
{% endcodeblock %}

## header=2
엑셀을 읽어들이다 보면 첫 번째 행이 header가 아닌 경우가 있습니다. 이 때 사용하는 option입니다. header=2의 의미는 3번째 행이 데이터의 header라는 의미입니다. 위의 코드를 고쳐보면 다음과 같습니다. 
{% codeblock lang:python %}
import pandas as np
BASE_PATH = "d:/dev"
FILE_NAME = "abc.xlsx"

from os.path import join

df = pd.read_excel(join(BASE_PATH, FILE_NAME),header=2)
df.head()
{% endcodeblock %}

## sheetname=None
Excel파일의 sheet가 여러 개인 경우 sheetname=None을 주고 read_excel을 실행시키면 엑셀파일 내의 모든 sheet를 읽어서 DataFrame으로 이루어진 Dict 객체를 반환합니다. 코드들 고쳐보겠습니다.
{% codeblock lang:python %}
import pandas as np
BASE_PATH = "d:/dev"
FILE_NAME = "abc.xlsx"

from os.path import join

df = pd.read_excel(join(BASE_PATH, FILE_NAME), sheetname=None)
df.head()
{% endcodeblock %}

## Related Posts

{% blockquote pandas 0.22.0 documentation https://pandas.pydata.org/pandas-docs/stable/10min.html#min 10 Minutes to pandas %} 10분 만에 익히는 pandas 기본{% endblockquote %}

{% blockquote Harish Garg https://www.dataquest.io/blog/excel-and-pandas/ Using Excel with pandas %} IMDB에서 제공하는 Excel로 pandas로 엑셀을 다루는 주요 Technique과 잘 설명하고  있습니다. {% endblockquote %}