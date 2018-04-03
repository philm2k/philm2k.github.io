---
title: Hexo블로그 관리:두개의 Repositories로 배포와 운영을 관리하기(Window) - 다소 불완전하지만...
categories:
  - Tips & Refs
  - Tips
tags:
  - hexo
date: 2018-03-18 13:17:04
thumbnail:
---

# 고민의 시작

Hexo로 github에 blog를 만든 것 까지는 좋았습니다. 그런데 형편에 따라 오늘은 집에 있는 PC에서 블로그 글을 작성하다가, 내일은 상암동에 있는 개발실의 PC에서 작업하고 하는 일이 있었습니다.  
hexo g -d로 배포하는 경우 "githubid".github.io repository에는 생성된 정적인 페이지만 배포가 되기 때문에 xxx.md로 작성된 post나 page는 별도의 저장소에 수작업으로 관리하게 되었습니다.  
이거 해보시면 압니다.  무지 불편하거든요 ^^

결론은 아래 방법도 완전하지는 않네요... 다른 PC에서 동일하게 하니 오류가 있었습니다.
결국 hueman의 _config.yml이 받아지지 않았더군요... 다른 곳에 두었던 것을 복사해 주니 해결은 됐는데... 찝찝합니다.

## 절차는 다음과 같습니다.
1. 블로그 소스를 저장할 repository를 만든다. 저의 경우는 hexo-blog-src로 했습니다.
2. github page를 배포할 reoository를 만든다. "githubid".github.io로 하는 것은 아시죠? ^^
3. 1번에서 만든 repository를 local에 clone한다.(GitHub Desktop을 사용하시면 나름 편리합니다.)  
4, 5번은 다음을 참고하시기 바랍니다.
{% blockquote Eric Han http://futurecreator.github.io/2016/06/14/get-started-with-hexo/ 워드프레스보다 쉬운 Hexo 블로그 시작하기 %} Hexo로 블로그를 시작했습니다. 다른 분들의 링크는 다소 내용이 빠져있어서 좀 헷갈렸는데, 이 포스트는 한 방에 잘 되더군요. 참 쉽네요 ^^ {% endblockquote %}
4. local에 clone된 곳(예를 들어 c:\hexo-blog-src)으로 가서 hexo 블로그를 만든다. 예를 들면...
{% codeblock %}
c:\hexo-blog-src\hexo init
c:\hexo-blog-src\npm instal hexo-cli
... 등등
{% endcodeblock %}
5. hexo 배포를 위한 git 설정을 합니다.
6. 그리고 themes를 git clone 명령을 이용하신 경우 해당 theme 폴더에 가 보시면 .git 폴더를 삭제하여 일반폴더로 만들어 주셔야 합니다. theme을 여러개 사용하신 경우 theme 폴더마다 들어가서 해주셔야 합니다.
{% codeblock %}
c:\hexo-blog-src\cd themes
c:\hexo-blog-src\themes\cd hueman (예를 들어 hueman theme을 쓰신 경우)
c:\hexo-blog-src\themes\hueman\del .git
... 등등
{% endcodeblock %}
7. 마지막으로 hexo d를 이용해 배포를 하시는 경우 .deploy_git 폴더가 생성되는데 이 것을 .gitignore에 추가해 주셔야 합니다. 제일 상단에 추가해 주세요 ^^
{% codeblock .gitignore %}
.deploy_git/ <== 요기처럼 ^^
# Logs
logs
*.log
...
{% endcodeblock %}

## 소스관리와 배포관리
- 소스는 다음의 명령으로 git push
{% codeblock %}
c:\hexo-blog-src\git add .
c:\hexo-blog-src\git commit -m "commit message"
c:\hexo-blog-src\git push origin master
{% endcodeblock %}
- 페이지 배포는 hexo g -d로 해결

### 특이사항
포스트를 작성하고 배포하기 전에 소스 repository에 commit만 하고 sync를 하지 않은 상태에서는 hexo d가 알수 없는 오류가 납니다. 그러니, 블로그에 배포가 끝난 후에 commit하고 sync하는 것이 좋을 것 같습니다.

### Related Posts
