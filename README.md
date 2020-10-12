# Mask-recogniser-on-RaspberryPi
It can recognise if a people wear the mask with RaspberryPi camera.

# Train

1. Download the dataset from the url or use a spider to get some images of people with or without mask
2. Put the images to $PATH_ROOT/catch_face/all_mask_img ,put the txts to $PATH_ROOT/all_mask
3. The format of the txt label is same as yolo_txt(every image with a txt, and mark every face with class,x,y,w,h ,normalized)
4. Run $PATH_ROOT/mask-recognize-master/catch_face/Generate.py
5. Copy the folder train to $PATH_ROOT/mask-recognize-master/data/image
6. Run $PATH_ROOT/mask-recognize-master/data/Generate_TrainTXT.py
7. Run $PATH_ROOT/mask-recognize-master/train.py

*** please train the model on your PC instead of on RaspberryPi ***

