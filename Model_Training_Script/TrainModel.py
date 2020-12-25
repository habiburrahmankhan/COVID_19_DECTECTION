import tensorflow as tf
import numpy as np
from Model import *


classes = ["COVID+ve" ,"COVID-ve"]
image_size = 224
x_train_path = "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/train_images.npy"
y_train_path = "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/train_labels.npy"
x_test_path = "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/valid_images.npy"
y_test_path = "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/valid_labels.npy"
path = "/Users/habiburrahmankhan/X_Ray_COVID_19/COVID_19_Detection_Using_Ensemble_Learning_master/"


x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

inception = train_model(path, x_train, y_train,
                        x_test, y_test, model_name = "inception_v3",
                        epochs = 10, input_shape = (image_size,image_size,3),
                        classes = len(classes))

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, 
                           mode='min', restore_best_weights=True)
callbacks = [early_stop]

densenet = train_model(path, x_train, y_train,
                       x_test, y_test, model_name="densenet201",
                       epochs=10, input_shape = (image_size,image_size,3),
                       classes = len(classes),
                       callbacks = callbacks)


resnet = train_model(path, x_train, y_train,
                     x_test, y_test, model_name="resnet50_v2",
                     epochs=10, input_shape = (image_size,image_size,3),
                     classes = len(classes),
                     callbacks = callbacks)


