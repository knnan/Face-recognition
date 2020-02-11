import cv2

vcap = cv2.VideoCapture("rtsp://admin:Unique123@106.51.130.230:6969/Streaming/Channels/101")

while(1):

    ret, frame = vcap.read()
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)
