from tools import *
import cv2 as cv
import os
import numpy as np

dataset_path = os.getcwd() + '/dataset' 
faces,labels,classes = prepare_training_data(dataset_path)#labels is the name of the img , classes is the number of the img

face_recogniser = cv.face.LBPHFaceRecognizer_create()#create a face #recogniser
face_recogniser.train(faces,np.array(classes))
face_recogniser.save(os.getcwd()+'/model.xml')

