from konlpy.tag import Okt
import pandas as pd 
import os

# 현재 스크립트가 위치한 디렉토리 경로 가져오기
script_dir = os.path.dirname(os.path.realpath(__file__))

# stopwords.xlsx의 절대 경로 생성
file_path = os.path.join(script_dir, 'stopwords.xlsx')

# 엑셀 파일 읽기
df = pd.read_excel(file_path, header=None)

# 첫 번째 열에 있는 데이터를 set으로 변환
stopwords_set = set(df[0].dropna().values)

text="최근 들어 많은 사람들이 취미로 코딩을 배우기 시작했습니다. 특히 온라인 교육 플랫폼과 유튜브 등의 자원을 활용하여 쉽고 빠르게 기초를 배우고 있습니다. 코딩을 배우면 문제 해결 능력도 향상되고, 창의적 사고를 기를 수 있는 장점이 많습니다. 또한, 프로그래밍 언어를 배우면 다양한 분야에서 활용할 수 있어 실용적입니다."
okt = Okt()

tokens = okt.morphs(text)

result = [word for word in tokens if not word in stopwords_set]

# 결과 출력
print(result)