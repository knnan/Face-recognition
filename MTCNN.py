from PIL import Image
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import mtcnn
import numpy


print('MTCNN Version : ',mtcnn.__version__)

filename = './known_faces/Abdullah/ab_1.png'

# load image from file
image = Image.open(filename)
# convert to RGB, if needed
image = image.convert('RGB')
# convert to array
pixels = numpy.asarray(image)

# print(pixels)


# create the detector, using default weights
detector = mtcnn.MTCNN()
# detect faces in the image
results = detector.detect_faces(pixels)
print(results)

x1, y1, width, height = results[0]['box']
# bug fix
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1 + width, y1 + height


