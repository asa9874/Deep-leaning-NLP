import cv2

# 이미지를 컬러 모드로 읽음
img = cv2.imread('img/test.png', cv2.IMREAD_COLOR)

#높이 너비 채널 추출
height, width, channel = img.shape

# 이미지 사이즈 조절
img = cv2.resize(img, dsize=(width//4, height//4), interpolation=cv2.INTER_AREA)

#부분 자르기
#너비 400~600 , 높이 300~600 자르기
img2=img[400:600, 300:600].copy()


# 반전
# -: 상하좌우
# +: 좌우
# 0: 상하
img = cv2.flip(img, 0)

# 이미지를 표시
cv2.imshow('title', img)
cv2.imshow('sometitle', img2)

# 키 입력 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()
