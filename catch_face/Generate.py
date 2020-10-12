import os
import cv2 as cv

path_txt = os.getcwd() + '\\all_mask'
path_img = os.getcwd() + '\\all_mask_img'
new_path = os.getcwd() + '\\mask_face'

#make a folder of img which caught face
folder = os.path.exists(new_path)
if not folder:
    os.makedirs(new_path)

cnt_mask = 1 # counters for named the dealed imgs
cnt_nomask = 1

dim=(600,600) 
dim_face=(160,160)

for imgs in os.listdir(path_img):
    imgs_path = os.path.splitext(imgs)[0]

    flag=0

    for txts in os.listdir(path_txt): #find the mark of the jpg as iterator(mark is more than imgs)
        txts_path_temp = os.path.splitext(txts)[0]

        if imgs_path == txts_path_temp:
            txts_path = txts_path_temp
            flag = 1
            break
    
    with open(path_txt + '\\' + str(txts_path) + ".txt","r") as txt:#open the imgs of mark
        img = cv.imread(path_img + '\\' + imgs)

        img_resized = img#cv.resize(img,dim,interpolation=cv.INTER_AREA)
        
        width = img_resized.shape[1] #original width of img
        height = img_resized.shape[0] #original height of img

        #in opencv x is 1 and the y is 0 !!!!!!

        lines = txt.readlines()
        for line in lines:

            line_withoutspace = line.strip()
            data=line_withoutspace.split(" ")

            
            anchor_x = int(width * float(data[1]))
            anchor_y = int(height * float(data[2]))
            anchor_width = int(width * float(data[3]))
            anchor_height = int(height * float(data[4]))

            anchor_size = int(min(anchor_height,anchor_width)/2)


            faced = cv.resize(img_resized[(anchor_y-anchor_size):(anchor_y + anchor_size),(anchor_x-anchor_size):(anchor_x + anchor_size)],dim_face) # catch the face
            
            if data[0] == '0':
                file_name = str("nomask_%d.jpg"%(cnt_mask))
                cnt_mask+=1
            
            else:
                file_name = str("mask_%d.jpg"%(cnt_nomask))
                cnt_nomask+=1

            #print("%s is dealed successfully"%(imgs))
            cv.imwrite(new_path+'\\' + file_name,faced)
            