import cv2

# 이미지를 컬러 모드로 읽음
img = cv2.imread('img/test.png', cv2.IMREAD_COLOR)

#높이 너비 채널 추출
height, width, channel = img.shape

# 이미지 사이즈 조절
img = cv2.resize(img, dsize=(width//4, height//4), interpolation=cv2.INTER_AREA)

#흑백 -> sobel,laplacian
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
img2 = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)

#Canny
img2 = cv2.Canny(img, 100, 255)
 

# 이미지를 표시
cv2.imshow('title', img)
cv2.imshow('title', img2)
# 키 입력 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()
