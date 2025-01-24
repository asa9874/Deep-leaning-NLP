#기본 구조 정책없음


import gym  # OpenAI Gym 라이브러리 임포트
from time import sleep
import matplotlib.pyplot as plt  # 시각화를 위한 라이브러리

# 1. 환경 만들기
env = gym.make('CartPole-v1')  # CartPole 환경 생성
num_episodes = 2000  # 학습할 에피소드 수
rList = [] #보상 기록리스트

for i in range(num_episodes):
    # 2. 환경 초기화
    print(i ,"번째")
    state = env.reset()  # 초기 상태를 설정하고 반환
    rAll = 0  # 에피소드 내 보상 합 초기화
    done = False  # 환경 종료 여부 초기화

    while not done:
        # 3. 환경을 렌더링 (시각적으로 표시)
        env.render()

        # 4. 행동 선택: 무작위로 선택하거나 정책 기반으로 선택
        action = env.action_space.sample()  # 환경에서 가능한 행동 중 하나를 무작위로 선택

        # 5. 선택된 행동을 환경에 적용
        # - state: 현재 상태
        # - action: 수행할 행동
        # - reward: 주어진 보상
        # - done: 에피소드 종료 여부 (True/False)
        # - info: 추가적인 정보
        next_state, reward, done, info = env.step(action)

        # 6. 총 보상 업데이트
        rAll += reward
    # 7. 각 에피소드의 총 보상을 rList에 저장
    rList.append(rAll)

# 8. 환경 닫기
env.close()  # 환경 종료 시, 리소스를 반환하고 종료

# 9. 보상 시각화
plt.plot(rList)  # 각 에피소드의 총 보상을 그래프에 표시
plt.xlabel('Episode')  # x축: 에피소드
plt.ylabel('Total Reward')  # y축: 총 보상
plt.title('Learning Progress (CartPole-v1)')  # 그래프 제목
plt.show()  # 그래프 출력
