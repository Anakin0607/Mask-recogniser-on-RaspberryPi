# Mask-recogniser-on-RaspberryPi
It can recognise if a people wear the mask with RaspberryPi camera.

## Environment

```
Tensorflow==1.13.1
keras==2.1.5
opencv-contrib-python==4.4.0.44
opencv-python==4.4.0.44
```

## Train

1. Download the dataset from the url or use a spider to get some images of people with or without mask
2. Put the images to $PATH_ROOT/catch_face/all_mask_img ,put the txts to $PATH_ROOT/all_mask
3. The format of the txt label is same as yolo_txt(every image with a txt, and mark every face with class,x,y,w,h ,normalized)
4. Run $PATH_ROOT/mask-recognize-master/catch_face/Generate.py
5. Copy the folder train to $PATH_ROOT/mask-recognize-master/data/image
6. Run $PATH_ROOT/mask-recognize-master/data/Generate_TrainTXT.py
7. Run $PATH_ROOT/mask-recognize-master/train.py

**Please train the model on your PC instead of on RaspberryPi**

## Predict
How to set the envirment of RaspberryPi please read that https://zhuanlan.zhihu.com/p/264994466

Run $PATH_ROOT/mask-recognize-master/mask_recognize.py

## Reference
https://github.com/bubbliiiing/mask-recognize

# Face-recogniser

## Train

1. Use the $PATH_ROOT/face-recognise/photo.py to take photo to the target face 
2. Run $PATH_ROOT/face-recognise/train.py and then the model will be saved as model.xml

## Predict

Run $PATH_ROOT/face-recognise/Predict,
