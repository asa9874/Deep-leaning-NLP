import cv2

# 이미지를 컬러 모드로 읽음
img = cv2.imread('img/test.png', cv2.IMREAD_COLOR)

#높이 너비 채널 추출
height, width, channel = img.shape

# 이미지 사이즈 조절
img = cv2.resize(img, dsize=(width//4, height//4), interpolation=cv2.INTER_AREA)

# 흑백사진
#img2= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 반전
#img2 = cv2.bitwise_not(img)

#이진화
#100: 임계값-> 픽셀 값이 이 값보다 크면 255로 설정되고, 그렇지 않으면 0으로 설정
#255: 최대값-> 초과된 픽셀은 255로 설정
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

# 블러
# 블러링(kernel) 크기 (가로,세로크기),
# 앵커 포인트 (-1,-1) -> 중심
img2 = cv2.blur(img, (10, 10), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)


# 이미지를 표시
cv2.imshow('title', img)
cv2.imshow('sometitle', img2)

# 키 입력 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()
