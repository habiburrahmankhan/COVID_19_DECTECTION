import numpy as np 
import os 
import cv2 
import random 
import threading


def create_data(datadir,save_filename_images,save_filename_labels,categories,image_size):
    
    data = []
    for category in categories:
        print("Current Directory:",datadir)
        print("Current class:",category)
        path = os.path.join(datadir,category)
        class_num = categories.index(category)
        files = os.listdir(path)
        total = len(files)
        print("Total:",total)
        current = 1
        for img in files: 
            print("Getting:", current, "of", total)              
            try:
                img_array = cv2.imread(os.path.join(path,img))
                img_array = cv2.resize(img_array,(image_size,image_size))
                data.append([img_array,class_num])                
            except Exception as e:
                pass 
            current += 1        
        print()          
    random.shuffle(data) 
    images = []
    classes = []    
    current = 1
    for image, cls in data:      
      images.append(image)
      classes.append(cls)
      current += 1
    
    print()  
    images = np.array(images).reshape(-1,image_size,image_size,3)
    images = images/255.0
    classes = np.array(classes)    
    print("Preparing .npy files....")
    np.save(save_filename_images,images)
    np.save(save_filename_labels,classes)
    print("npy Files Ready")

TRAIN_DIR = input("Enter the path to the training images: ")
TEST_DIR = input("Enter the path to the testing images: ")
DESTINATION = input("Enter the path to save the npy files: ")
CLASSES = ["COVID+ve" ,"COVID-ve"]
IMAGE_SIZE = 224

train_image_path = DESTINATION + "/train_images.npy"   
train_labels_path = DESTINATION + "/train_labels.npy"   
test_image_path = DESTINATION + "/test_images.npy"
test_labels_path = DESTINATION + "/test_labels.npy"

b_tr_thread = threading.Thread(target=create_data,args=(TRAIN_DIR,train_image_path,train_labels_path,CLASSES,IMAGE_SIZE))
b_te_thread = threading.Thread(target=create_data,args=(TEST_DIR,test_image_path,test_labels_path,CLASSES,IMAGE_SIZE))

b_tr_thread.start()
b_te_thread.start()
b_tr_thread.join()
b_te_thread.join()