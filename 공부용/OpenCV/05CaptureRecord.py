import datetime
import cv2

# 비디오 파일 열기
capture = cv2.VideoCapture("./img/test.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
is_recording = False

while True:
    # 비디오의 마지막 프레임에 도달하면 처음으로 다시 설정
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.open("./img/test.mp4")

    ret, frame = capture.read()  # 프레임 읽기
    cv2.imshow("VideoFrame", frame)  # 프레임 표시

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)  # 33ms마다 프레임 갱신

    # 종료 (ESC 키)
    if key == 27: 
        break

    # 캡처 (CTRL+Z)
    elif key == 26: 
        print("Image Captured")
        cv2.imwrite("./img" + str(now) + ".png", frame)

    # 녹화 시작 (CTRL+X)
    elif key == 24:
        is_recording = True
        video_writer = cv2.VideoWriter("D:/" + str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print("Recording Started")

    # 녹화 중지 (CTRL+C)
    elif key == 3:
        is_recording = False
        video_writer.release()
        print("Recording Stopped")
        
    # 녹화 중일 때만 영상 저장
    if is_recording:
        print("Recording...")
        video_writer.write(frame)

# 자원 해제
capture.release()
cv2.destroyAllWindows()
