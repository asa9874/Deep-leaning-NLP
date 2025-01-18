import cv2
import numpy as np
# 이미지를 컬러 모드로 읽음
img = cv2.imread('img/test.png', cv2.IMREAD_COLOR)

#높이 너비 채널 추출
height, width, channel = img.shape

# 이미지 사이즈 조절
img = cv2.resize(img, dsize=(width//4, height//4), interpolation=cv2.INTER_AREA)


#HSV 변경
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV 분리
imgh, imgs, imgv = cv2.split(img2)
# cv2.imshow('titleH', imgh)
# cv2.imshow('titleS', imgs)
# cv2.imshow('titleV', imgv)


# 파란영역 검출
lower_blue = np.array([100, 50, 50])  # 파란색의 낮은 범위
upper_blue = np.array([140, 255, 255])  # 파란색의 높은 범위
blueH = cv2.inRange(img2, lower_blue, upper_blue)  # 파란색 영역 검출


# 하얀영역 검출
lower_white = np.array([0, 0, 200])  # 하얀색의 낮은 범위
upper_white = np.array([180, 50, 255])  # 하얀색의 높은 범위
whiteH= cv2.inRange(img2, lower_white, upper_white)

BlueAndWhiteH=cv2.addWeighted(blueH, 1.0, whiteH, 1.0, 0.0)


BlueAndWhite = cv2.bitwise_and(img2, img2, mask=BlueAndWhiteH)  # 파란색 영역 마스크 적용
BlueAndWhite = cv2.cvtColor(BlueAndWhite, cv2.COLOR_HSV2BGR)  # HSV를 BGR로 변환

cv2.imshow('titleBlueAndWhite', BlueAndWhite)


# 병합
img2_merged = cv2.merge((imgh, imgs, imgv))  
img2_merged = cv2.cvtColor(img2_merged, cv2.COLOR_HSV2BGR)
cv2.imshow('titleMerge', img2_merged)


# 키 입력 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()
