import tensorflow as tf 
import numpy as np
from Ensembling import *
from sklearn.metrics import classification_report, confusion_matrix

x_test_path = "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/test_images.npy" #input("Enter path to test labels: ")
y_test_path = "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/test_labels.npy" #input("Enter path to test labels: ") # 

inception_path ="/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/GUI/saveModels/inception_v3.h5" # input("Enter path to Inception Model: ") 
resnet_path =   "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/GUI/saveModels/resnet50_v2.h5" #  input("Enter path to Resnet Model: ")  
densenet_path =  "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/GUI/saveModels/densenet201.h5" #  input("Enter path to DenseNet Model: ") 

x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

image_size = 224

inception_model = tf.keras.models.load_model(inception_path)
resnet_model = tf.keras.models.load_model(resnet_path)
densenet_model = tf.keras.models.load_model(densenet_path)

models = [densenet_model,resnet_model,inception_model]

w = generate_weights(x_test,y_test,models)[0] #generating weights
print("Weights: ", w)

predictions = []
for i in range(len(x_test)):
  pred = ensemble(x_test[i].reshape(-1,image_size,image_size,3),w,models)
  predictions.append(pred)

print("Accuracy: ",accuracy(predictions,y_test))

y_pred = np.argmax(np.array(predictions), axis=1)

print("The classification report: ")
print(classification_report(y_pred=y_pred, y_true=y_test))
print()
print("Confusion Matrix: ")
print(confusion_matrix(y_pred=y_pred, y_true=y_test))



# /Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/GUI/saveModels20/resnet50_v2.h5