import cv2 as cv

Video = cv.VideoCapture(0)

while True:
    ret,frame = Video.read()
    cv.imshow("capture",frame)
    cv.waitKey(33)

Video.release()
cv.destroyAllWindows()