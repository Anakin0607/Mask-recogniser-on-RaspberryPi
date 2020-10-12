import os
import cv2 as cv
from train import labels
from tools import detect_face

def draw_rectangle(img,rect):
    (x,y,w,h) = rect
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

def put_text(img,text,x,y):
    cv.putText(img,text,(x,y),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cap = cv.VideoCapture(0)

while True:
    ret,img = cap.read()

    cv.imshow("figure",img)
    cv.waitKey(33)
    face,rect,flag = detect_face(img)
    if flag == 0:
        continue
 
    face_recogniser = cv.face.LBPHFaceRecognizer_create()
    face_recogniser.read(os.getcwd()+'/model.xml')

    label = face_recogniser.predict(face)

    label_text = str(labels[label[0]])# get the name of the face from num
    draw_rectangle(img,rect)
    put_text(img,label_text,int(rect[0]),int(rect[1]))

    cv.imshow("figure",img)
    cv.waitKey(100)
