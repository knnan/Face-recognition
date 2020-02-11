from MTCNN_face_detection import extract_face


from matplotlib import pyplot
from os import listdir

# specify folder to plot
folder = '/home/knnan/Development/face_recognition/known_faces/Hameem/'
i = 1
# enumerate files
for filename in listdir(folder):
	# path
	path = folder + filename
	# get face
	face = extract_face(path)
	print(i, face.shape)
	# plot
	pyplot.subplot(2, 7, i)
	pyplot.axis('off')
	pyplot.imshow(face)
	i += 1
pyplot.show()
