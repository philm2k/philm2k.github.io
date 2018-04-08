
### 1. glob.glob을 이용하여 파일목록 가져오기


```python
from glob import glob
lst_files = glob("D:/HobbyDev/02.python Data Analysis/DataScienceHandbook/*.ipynb")
lst_files[:5]
```




    ['D:/HobbyDev/02.python Data Analysis/DataScienceHandbook\\00.00-Preface.ipynb',
     'D:/HobbyDev/02.python Data Analysis/DataScienceHandbook\\01.00-IPython-Beyond-Normal-Python.ipynb',
     'D:/HobbyDev/02.python Data Analysis/DataScienceHandbook\\01.01-Help-And-Documentation.ipynb',
     'D:/HobbyDev/02.python Data Analysis/DataScienceHandbook\\01.02-Shell-Keyboard-Shortcuts.ipynb',
     'D:/HobbyDev/02.python Data Analysis/DataScienceHandbook\\01.03-Magic-Commands.ipynb']



## 2. getctime을 이용하여 가장 최근에 생성된 파일명 찾기


```python
from os.path import getctime
latest_file = max(lst_files, key=getctime)
latest_file
```




    'D:/HobbyDev/02.python Data Analysis/DataScienceHandbook\\03.12-Performance-Eval-and-Query.ipynb'


