import subprocess

# 입력 파일과 출력 파일 경로 설정
input_model = "kor_w2v"  # 모델 파일 경로
output_model = "kor_w2v_output"  # 출력 파일 경로

try:
    # word2vec2tensor 명령어를 subprocess로 실행
    subprocess.run(['python', '-m', 'gensim.scripts.word2vec2tensor', '--input', input_model, '--output', output_model], check=True)
    print("TensorFlow 형식으로 변환 완료!")

except subprocess.CalledProcessError as e:
    print(f"명령어 실행 중 오류 발생: {e}")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 'kor_w2v' 경로와 파일 이름을 확인하세요.")
except Exception as e:
    print(f"예기치 않은 오류 발생: {e}")