import warnings
warnings.simplefilter(action='ignore')

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
svcmodel = SVC(kernel='linear', probability=True)


# from test_opencv import test_opencv.getFace, drayBOX, usePIL
import test_opencv

# example of loading the keras facenet model
from keras.models import load_model

model_path = '/home/knnan/Development/face_recognition/Facenet/keras-facenet/model/facenet_keras.h5'



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









# load the model
model = load_model(model_path)
print('Facenet Model Loaded')
# summarize input and output shape
print('INPUT FORMAT  : ', model.inputs)
print('OUTPUT FORMAT : ',model.outputs)





def save_dataset_embeddings(faces_numpy_file_path,face_embeddings_file_name):
	data = load(faces_numpy_file_path)
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
	# convert each face in the train set to an embedding
	newTrainX = list()
	for face_pixels in trainX:
		embedding = get_embedding(model, face_pixels)
		newTrainX.append(embedding)
	newTrainX = asarray(newTrainX)
	print(newTrainX.shape)
	# convert each face in the test set to an embedding
	newTestX = list()
	for face_pixels in testX:
		embedding = get_embedding(model, face_pixels)
		newTestX.append(embedding)
	newTestX = asarray(newTestX)
	print(newTestX.shape)
	# save arrays to one file in compressed format
	# /home/knnan/Development/face_recognition/Facenet/5-celebrity-faces-embeddings.npz : face_embeddings_file_name
	savez_compressed('/home/knnan/Development/face_recognition/Facenet/' +face_embeddings_file_name+'.npz' , newTrainX, trainy, newTestX, testy)



def predict(faces_data_file, face_embeddings_file):
	

	face_data = load(faces_data_file)
	testX_faces = face_data['arr_2']
	data = load(face_embeddings_file)
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
	print(testX)


	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)



	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)



	# fit model
	model = SVC(kernel='linear', probability=True)
	model.fit(trainX, trainy)

	# predict
	yhat_train = model.predict(trainX)
	yhat_test = model.predict(testX)
	# score
	score_train = accuracy_score(trainy, yhat_train)
	score_test = accuracy_score(testy, yhat_test)
	# summarize
	print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


	for index in [i for i in range(testX.shape[0])]:

		# test model on a random example from the test dataset
		selection = choice([i for i in range(testX.shape[0])])
		selection =index
		random_face_pixels = testX_faces[selection]
		random_face_emb = testX[selection]
		random_face_class = testy[selection]
		random_face_name = out_encoder.inverse_transform([random_face_class])
		# prediction for the face
		samples = expand_dims(random_face_emb, axis=0)
		yhat_class = model.predict(samples)
		yhat_prob = model.predict_proba(samples)
		# get name
		class_index = yhat_class[0]
		class_probability = yhat_prob[0,class_index] * 100
		predict_names = out_encoder.inverse_transform(yhat_class)
		print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
		print('Expected: %s' % random_face_name[0])
		# plot for fun
		pyplot.imshow(random_face_pixels)
		title = '%s (%.3f)' % (predict_names[0], class_probability)
		pyplot.title(title)
		pyplot.show()

	# for i in range(len(testX)):
	# 	test_face = testX_faces[i]
	# 	test_embedding = testX[i]
	# 	test_face_class = testy
	# 	test_face_class = testy[i]
	# 	test_face_name = out_encoder.inverse_transform([test_face_class])
	# 	samples = expand_dims(test_embedding, axis=0)
	# 	yhat_class = model.predict(samples)
	# 	yhat_prob = model.predict_proba(samples)
	# 	# get name
	# 	class_index = yhat_class[0]
	# 	class_probability = yhat_prob[0,class_index] * 100
	# 	predict_names = out_encoder.inverse_transform(yhat_class)

	# 	print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
	# 	print('Expected: %s' % test_face_name[0])

	# 	pyplot.subplot(2, 7, i)
	# 	pyplot.axis('off')
	# 	pyplot.imshow(face)
	# 	i += 1










def opencv_predict(face_embeddings_file,prediction_image_path,expected_name):

	#LOAD FACENET MODEL
	
	model_path = '/home/knnan/Development/face_recognition/Facenet/keras-facenet/model/facenet_keras.h5'
	model = load_model(model_path)


	opencv_face_pixels = test_opencv.getFace(prediction_image_path)
	original_image = opencv_face_pixels[1]
	cord = opencv_face_pixels[2]
	# opencv_face_pixels = usePIL(prediction_image_path)
	print(opencv_face_pixels)
	face_embedding = get_embedding(model, opencv_face_pixels[0])

	data = load(face_embeddings_file)
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)
	
	face_embedding = in_encoder.transform([face_embedding])
	face_embedding = face_embedding[0]
	
	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)



	# fit model
	model = SVC(kernel='linear', probability=True)
	model.fit(trainX, trainy)

	# predict
	yhat_train = model.predict(trainX)
	yhat_test = model.predict(testX)
	# score
	score_train = accuracy_score(trainy, yhat_train)
	score_test = accuracy_score(testy, yhat_test)
	# summarize
	print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


	# DO PREDICTION
	samples = expand_dims(face_embedding, axis=0)
	yhat_class = model.predict(samples)
	yhat_prob = model.predict_proba(samples)
	class_index = yhat_class[0]
	class_probability = yhat_prob[0,class_index] * 100
	predict_names = out_encoder.inverse_transform(yhat_class)
	print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
	print('Expected: %s' % expected_name)
	test_opencv.drawBOX(original_image,cord[0],cord[1],cord[2],cord[3],predict_names[0])
	


def loadFacenet_model(model_path='/home/knnan/Development/face_recognition/Facenet/keras-facenet/model/facenet_keras.h5'):
	model = load_model(model_path)
	return model

def opencv_predict_video(face_embeddings_data,prediction_image_path,expected_name):

	#LOAD FACENET MODEL
	
	model_path = '/home/knnan/Development/face_recognition/Facenet/keras-facenet/model/facenet_keras.h5'
	model = load_model(model_path)


	opencv_face_pixels = test_opencv.getFace(prediction_image_path)
	original_image = opencv_face_pixels[1]
	cord = opencv_face_pixels[2]
	# opencv_face_pixels = usePIL(prediction_image_path)
	print(opencv_face_pixels)
	face_embedding = get_embedding(model, opencv_face_pixels[0])

	data = face_embeddings_data
	trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
	print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)
	
	face_embedding = in_encoder.transform([face_embedding])
	face_embedding = face_embedding[0]
	
	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)



	# fit model
	global svcmodel
	svcmodel.fit(trainX, trainy)

	# predict
	# yhat_train = svcmodel.predict(trainX)
	# yhat_test = svcmodel.predict(testX)
	# score
	# score_train = accuracy_score(trainy, yhat_train)
	# score_test = accuracy_score(testy, yhat_test)
	# summarize
	# print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


	# DO PREDICTION
	samples = expand_dims(face_embedding, axis=0)
	yhat_class = svcmodel.predict(samples)
	yhat_prob = svcmodel.predict_proba(samples)
	class_index = yhat_class[0]
	class_probability = yhat_prob[0,class_index] * 100
	predict_names = out_encoder.inverse_transform(yhat_class)
	# print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
	# print('Expected: %s' % expected_name)
	return predict_names[0]
	# test_opencv.drawBOX(original_image,cord[0],cord[1],cord[2],cord[3],predict_names[0])








save_dataset_embeddings('/home/knnan/Development/face_recognition/Facenet/all_faces.npz','all_faces_embeddings')


# NORMAL PREDICTION
# predict('/home/knnan/Development/face_recognition/Facenet/Custom_faces.npz','/home/knnan/Development/face_recognition/Facenet/Custom_face_embeddings.npz')

# OPENCV PREDICTION

# image_to_predict = '/home/knnan/Development/face_recognition/unknown_faces/Abdullah/abdullah2.png'
# opencv_predict('/home/knnan/Development/face_recognition/Facenet/Custom_face_embeddings.npz',image_to_predict,'Abdullah')
