import cv2 as cv
import os
import numpy as np

def detect_face(img):

    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_gray,scaleFactor = 1.2,minNeighbors = 5)

    if len(faces) == 0: # if there is no face in the figure return none
        return None,None,0
    
    (x,y,w,h) = faces[0] #only detect one face

    return img_gray[y:(y+w),x:(x+h)],faces[0],1 #return the img and axis

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    classes =[]

    cnt = -1
    for dir_name in dirs:#iterate every kinds of data

        labels.append(dir_name)# create the connect of the num and the name

        cnt+=1
        label = cnt # opencv only can read the number label
        img_path = os.path.join(data_folder_path,dir_name)
        sub_img_names = os.listdir(img_path)#the label of the data

        for img_name in sub_img_names:

            img = cv.imread(os.path.join(img_path,img_name))
            face,rect,flag = detect_face(img)#detect the face
            if face is not None:
                faces.append(face)
                classes.append(label)
    
    return faces,labels,classes #labels is the name of the img , classes is the number of the img
