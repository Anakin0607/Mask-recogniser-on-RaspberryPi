import cv2 as cv
import os

path = os.getcwd() + '/dataset'

print("Input your name")

sub_path = input()
data_path = path + '/' + sub_path

if not os.path.exists(data_path):
    os.mkdir(data_path)

cap = cv.VideoCapture(0)

print("Please press space if you're prepared")

flag = 0

for i in range(0,30):#get 30 photos

    ret,img = cap.read()
    cv.imshow("figure",img)
    if flag == 0:
        while True:# give person a time to prepare a pose
            ret,img = cap.read()
            cv.imshow("figure",img)
            if cv.waitKey(33) == 32:
                flag = 1
                break
    #when you prepared, please push the space
    cv.imshow("figure",img)    
    cv.waitKey(300)#get photo every 300ms

    img_name = sub_path + "_%d.jpg"%(i)
    cv.imwrite(data_path + '/' + img_name,img)
