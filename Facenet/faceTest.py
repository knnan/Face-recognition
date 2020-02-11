import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cv2
from scipy.spatial import distance
import numpy as np
import os
from time import perf_counter
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import time
from timeit import default_timer as timer
import math
from mtcnn.mtcnn import MTCNN
from random import choice
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from os import listdir
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from keras.models import load_model
from faced import FaceDetector
from faced.utils import annotate_image







def display_multi_images(images_array):
    f = pyplot.figure(figsize=(20, 20))
    columns = 3
    rows = 1
    image_index = 0
    for i in range(1, columns*rows +1):
        
        if(image_index+1 > len(images_array)):
            break
        f.add_subplot(rows, columns, i)
        pyplot.imshow(images_array[image_index])
        image_index += 1
     
    pyplot.show()




def get_multi_embeddings(model, face_pixels_array):
    embeddings_array = []
    for face_pixels in face_pixels_array:
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        # return back face embedding
        embeddings_array.append(yhat[0])
        
    print('No of embeddings : ',len(embeddings_array))
    return in_encoder.transform(embeddings_array)
    

def euc(a,b):
    dst = distance.euclidean(a, b)
    return dst















print("Program started")
print("executing")

model_path = '/home/knnan/Development/face_recognition/Facenet/keras-facenet/model/facenet_keras.h5'
face_embeddings_file = '/home/knnan/Development/face_recognition/Facenet/Custom_face_embeddings.npz'

# Global variables and definitions
model = load_model(model_path)
print("Model has Loaded")
pre_embeddings_data = load(face_embeddings_file)


face_detector = FaceDetector()


