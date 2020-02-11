import cv2

from faced import FaceDetector
from faced.utils import annotate_image

face_detector = FaceDetector()
print()
img_path = '/home/knnan/Development/face_recognition/unknown_faces/Abdullah/abdullah2.png'
img = cv2.imread(img_path)
rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

# Receives RGB numpy image (HxWxC) and
# returns (x_center, y_center, width, height, prob) tuples. 
bboxes = face_detector.predict(rgb_img)

# Use this utils function to annotate the image.
ann_img = annotate_image(img, bboxes)
print(bboxes)

cv2.imshow('image',ann_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
