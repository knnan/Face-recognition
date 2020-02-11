from MTCNN_face_detection import load_dataset
from numpy import savez_compressed


training_faces_folder='/home/knnan/Development/face_recognition/known_faces/'
testing_faces_folder='/home/knnan/Development/face_recognition/unknown_faces/'
# load train dataset
trainX, trainy = load_dataset(training_faces_folder)
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset(testing_faces_folder)
print(testX.shape, testy.shape)
# save arrays to one file in compressed format
savez_compressed('/home/knnan/Development/face_recognition/Facenet/all_faces.npz', trainX, trainy, testX, testy)


