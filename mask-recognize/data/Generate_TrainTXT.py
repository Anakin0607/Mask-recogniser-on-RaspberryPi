import os 
classes = ["mask","nomask"]
path = os.getcwd()
with open(path+'/train.txt','w') as f:
    after_generate = os.listdir(path+cd"/image/train")
    for image in after_generate:
    	if image.endswith("jpg"):
        	f.write(image + ";" + str(classes.index(image.split("_")[0]))+ "\n")
