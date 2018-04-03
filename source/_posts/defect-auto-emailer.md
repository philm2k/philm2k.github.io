---
title: 엑셀가공하여 자동 이메일 발송하기
categories:
  - Data Science
  - Pandas
tags:
  - Excel
  - Outlook
  - Automation
date: 2018-03-28 00:17:42
thumbnail:
---

ㅋㅋㅋ 오늘은 제가 하는 일에 pandas와 python을 접목시켜 직원들을 괴롭히는(?) 자동화 프로그램을 하나 작성했습니다.

# 프로그램 개요

- 테스트관리도구에서 selenium을 이용해서 결함내역을 scrapping

- pandas의 read_excel로 읽어들여서 필요한 파일만 dataframe에 저장

- PC의 outlook을 실행

- 2번에서 생성한 DataFrame을 Filtering해서 각 담당자들에게 자신에게 할당된 해결되지 않은 결함을 자동으로 메일 발송

소스코드는 아래에 곧 공개하겠습니다. ^^

## 1. selenium을 이용해서 web scrapping

## 2. pd.read_excel로 필요한 column만 DataFrame에 Load

## 3. PC의 outlook 실행

{% post_link send-email-with-outlook %} 참고
{% post_link python-how-to-C %}
## 4. 2번에서 생성한 DataFrame을 가공해서 각 담당자에게 자동 이메일 발송

### Related Postsㅗㄷㅌ