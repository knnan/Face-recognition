import cv2
import numpy as np
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import time
from mtcnn.mtcnn import MTCNN
# from keras_facenet import get_embedding

img = '/home/knnan/Development/face_recognition/unknown_faces/Abdullah/abdullah2.png'



def usePIL(image_path):
	required_size=(160, 160)
	image = Image.open(image_path)
	# convert to RGB, if needed
	image = image.convert('RGB')
	# convert to array
	pixels = asarray(image)
	print(pixels)

	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


	cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

	cv2.waitKey(0)
	return face_array

def userOPENCV(image_path):
	required_size=(160, 160)
	readimage = cv2.imread(image_path)
	rgb_image = cv2.cvtColor(readimage, cv2.COLOR_BGR2RGB)
	# cv2.imshow('BGR Image', rgb_image)
	# cv2.waitKey(0)

	pixels = rgb_image

	print(pixels)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	print(x1, x2, y1, y2)
	
	left = x1
	top = y1
	right = x2
	bottom = y2
	cv2.rectangle(readimage, (left, top), (right, bottom), (0, 0, 255), 2)
	# Draw a label with a name below the face
	cv2.rectangle(readimage, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	font = cv2.FONT_HERSHEY_DUPLEX
	name ='Abdullah'
	cv2.putText(readimage, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	cv2.imshow('BGR Image', readimage)
	cv2.waitKey(0)
	return face_array


def getFace(image_path, required_size=(160, 160)):
	original_image = cv2.imread(image_path)
	rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	# cv2.imshow('BGR Image', rgb_image)
	# cv2.waitKey(0)

	pixels = rgb_image
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	# bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# extract the face
	coordinates = [x1,y1,x2,y2]
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return [face_array,original_image,coordinates]






def drawBOX(original_image, left, top, right, bottom, name):
	# Draw a rectangle around the face
	cv2.rectangle(original_image, (left, top), (right, bottom), (0, 0, 255), 2)
	# Draw a label with a name below the face
	cv2.rectangle(original_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	font = cv2.FONT_HERSHEY_DUPLEX
	cv2.putText(original_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	cv2.imshow('BGR Image', original_image)
	cv2.waitKey(0)
	# return original_image



	


	
	

# userOPENCV(img)
# pyplot.subplot(2, 7, 1)
# pyplot.axis('off')
# pyplot.imshow(usePIL(img))
# pyplot.subplot(2, 7, 2)
# pyplot.axis('off')
# pyplot.imshow(userOPENCV(img))




# pyplot.show()



