---
title: outlook으로 이메일 보내기
categories:
  - Python
tags:
  - H2
  - Send Email
  - Outlook
date: 2018-03-29 23:11:42
thumbnail:
---

stackoverflow등 인터넷을 뒤지면서 찾은 python으로 email 보내는 소스입니다.
pywin을 사용할 수도 있는 것 같은데 아래 소스는 win32를 이용한 방법입니다.

{% codeblock lang:python %}
import win32com.client as win32
import psutil
import subprocess

def send_email(subject, body, recipients, cc, auto=True):
    """
    이메일 보내는 함수
    Auto=True인 경우 바로 메일 발송
    Auto=False인 경우 메일 보내기 위한 편지 화면을 Display
    """
    outlook = wind32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)

    if type(recipients) == list:
        mail.To = ";".join(recipients)
    else:
        mail.To = recipients
    
    if type(cc) == list:
        mail.CC = ";".join(cc)
    else:
        mail.CC = cc

    mail.Subject = subject
    mail.HtmlBody = body

    if auto:
        mail.send
    else:
        mail.Display(True)

def open_outlook():
    """
    오피스 버전에 따라 위치는 조금씩 다름
    """
    try:
        subprocess.call(['c:\Program Files\Microsoft Office\Office14\Outlook.exe'])
        os.system(['c:\Program Files\Microsoft Office\Office14\Outlook.exe'])
    except:
        print("Outlook open 실패!!!")

# outlook이 실행중이면 flag = 1, 아니면 flag = 0
for item in psutil.pids():
    p = psutil.Process(item)
    if p.name() == "Outlook.com":
        flag = 1
        break
    else:
        flag = 0

if (flag != 1):
    open_outlook()

send_email("제목","이메일 발송 테스트","gdhong@abcdef.com", "wcjeon@abcdef.com")

{% endcodeblock %}

### Related Posts