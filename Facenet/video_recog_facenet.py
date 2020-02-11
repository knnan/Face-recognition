import cv2
import numpy as np
import os
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import time
from timeit import default_timer as timer
import math
from mtcnn.mtcnn import MTCNN
# import warnings
# warnings.simplefilter(action='ignore')
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
from faced import FaceDetector
from faced.utils import annotate_image

# import keras
# config = tf.ConfigProto(device_count={"CPU": 4})
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))



# MULTI CORE CODE
from keras import backend as K
import tensorflow as tf
NUM_PARALLEL_EXEC_UNITS = 4
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"




from keras.models import load_model


def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	# return back face embedding
	return yhat[0]


def get_multi_embeddings(model, face_pixes_array):
	embeddings_array = []
	for face_pixels in face_pixes_array:
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
		return yhat[0]





def recog_video():

	model_path = '/home/knnan/Development/face_recognition/Facenet/keras-facenet/model/facenet_keras.h5'
	model = load_model(model_path)
	face_embeddings_file = '/home/knnan/Development/face_recognition/Facenet/Custom_face_embeddings.npz'
	# celebrties face_embeddings file
	face_embeddings_file = '/home/knnan/Development/face_recognition/Facenet/Custom_face_embeddings.npz'

	data = load(face_embeddings_file)
	face_detector = FaceDetector()
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)
	
	
	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)
	svcmodel = SVC(kernel='linear', probability=True)
	svcmodel.fit(trainX, trainy)

	video_capture = cv2.VideoCapture('/home/knnan/Development/face_recognition/videos/ab_ha.mp4')
	print("FRAMES/sec : ",cv2.CAP_PROP_FPS)
	frameRate = video_capture.get(5)  #frame rate
	print(frameRate)
	required_size = (160, 160)
	frame_count=1
	while (video_capture.isOpened()):
		
			
		start = timer()
		ret, frame = video_capture.read()
		if (ret != True):
			break
		original_image = frame
		rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		pixels = rgb_image




		# # create the detector, using default weights
		# detector = MTCNN()
		# # detect faces in the image
		# results = detector.detect_faces(pixels)
		# # extract the bounding box from the first face
		# x1, y1, width, height = results[0]['box']

		
		# faced mode detection
		bbox = face_detector.predict(pixels)
		if (len(bbox) < 1):
			continue
		x, y, width, height,p = bbox[0]
		#end faced mode detection

		end = timer()
		print('Face detections time : ', end - start)
		
		x1 = int(x - width/2)
		y1 = int(y - height/2)
		x2 = int(x + width/2)
		y2 = int(y + height/2)


		# bug fix
		# x1, y1 = abs(x1), abs(y1)
		# x2, y2 = x1 + width, y1 + height
		# extract the face
		coordinates = [x1,y1,x2,y2]
		face = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)

		
		start = timer()
		face_embedding = get_embedding(model, face_array)
		end = timer()
		print('Face Embedding time : ', end - start)
		start = timer()
		
		face_embedding = in_encoder.transform([face_embedding])
		face_embedding = face_embedding[0]
		samples = expand_dims(face_embedding, axis=0)
		yhat_class = svcmodel.predict(samples)
		yhat_prob = svcmodel.predict_proba(samples)
		class_index = yhat_class[0]
		class_probability = yhat_prob[0,class_index] * 100
		predict_names = out_encoder.inverse_transform(yhat_class)
		guessed_name = predict_names[0]
		end = timer()
		print('prediction time : ', end - start)


		left = x1
		top = y1
		right = x2
		bottom = y2
		cv2.rectangle(original_image, (left, top), (right, bottom), (0, 0, 255), 2)
		# Draw a label with a name below the face
		cv2.rectangle(original_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(original_image, guessed_name + '_' + str(class_probability), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		original_image = cv2.resize(original_image,(800,800))
		cv2.imshow('Video', original_image)



		print('Frame no : ', frame_count)
		print('predicted person : ',guessed_name )
		print('class_probability : ',class_probability )

		frame_count = frame_count + 1



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	video_capture.release()
	cv2.destroyAllWindows()



def recog_multi_video():
	model_path = '/home/knnan/Development/face_recognition/Facenet/keras-facenet/model/facenet_keras.h5'
	model = load_model(model_path)
	face_embeddings_file = '/home/knnan/Development/face_recognition/Facenet/Custom_face_embeddings.npz'
	# celebrties face_embeddings file
	face_embeddings_file = '/home/knnan/Development/face_recognition/Facenet/Custom_face_embeddings.npz'

	data = load(face_embeddings_file)
	face_detector = FaceDetector()
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)
	
	
	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)
	svcmodel = SVC(kernel='linear', probability=True)
	svcmodel.fit(trainX, trainy)
	

	video_capture = cv2.VideoCapture('/home/knnan/Development/face_recognition/videos/ab_ha.mp4')
	print("FRAMES/sec : ",cv2.CAP_PROP_FPS)
	frameRate = video_capture.get(5)  #frame rate
	print(frameRate)
	required_size = (160, 160)
	frame_count=1
	while (video_capture.isOpened()):
		
			
		start = timer()
		ret, frame = video_capture.read()
		if (ret != True):
			break
		original_image = frame
		rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		pixels = rgb_image




		# # create the detector, using default weights
		# detector = MTCNN()
		# # detect faces in the image
		# results = detector.detect_faces(pixels)
		# # extract the bounding box from the first face
		# x1, y1, width, height = results[0]['box']

		
		# faced mode detection
		bbox = face_detector.predict(pixels)
		if (len(bbox) < 1):
			continue
		x, y, width, height,p = bbox[0]
		#end faced mode detection

		end = timer()
		print('Face detections time : ', end - start)
		
		x1 = int(x - width/2)
		y1 = int(y - height/2)
		x2 = int(x + width/2)
		y2 = int(y + height/2)


		# bug fix
		# x1, y1 = abs(x1), abs(y1)
		# x2, y2 = x1 + width, y1 + height
		# extract the face
		coordinates = [x1,y1,x2,y2]
		face = pixels[y1:y2, x1:x2]
		# resize pixels to the model size
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = asarray(image)

		
		start = timer()
		face_embedding = get_embedding(model, face_array)
		end = timer()
		print('Face Embedding time : ', end - start)
		start = timer()
		
		face_embedding = in_encoder.transform([face_embedding])
		face_embedding = face_embedding[0]
		samples = expand_dims(face_embedding, axis=0)
		yhat_class = svcmodel.predict(samples)
		yhat_prob = svcmodel.predict_proba(samples)
		class_index = yhat_class[0]
		class_probability = yhat_prob[0,class_index] * 100
		predict_names = out_encoder.inverse_transform(yhat_class)
		guessed_name = predict_names[0]
		end = timer()
		print('prediction time : ', end - start)


		left = x1
		top = y1
		right = x2
		bottom = y2
		cv2.rectangle(original_image, (left, top), (right, bottom), (0, 0, 255), 2)
		# Draw a label with a name below the face
		cv2.rectangle(original_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(original_image, guessed_name + '_' + str(class_probability), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		original_image = cv2.resize(original_image,(800,800))
		cv2.imshow('Video', original_image)



		print('Frame no : ', frame_count)
		print('predicted person : ',guessed_name )
		print('class_probability : ',class_probability )

		frame_count = frame_count + 1



		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	video_capture.release()
	cv2.destroyAllWindows()




recog_video()
