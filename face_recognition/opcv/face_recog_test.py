import face_recognition

known_image_aaron = face_recognition.load_image_file("/home/knnan/Development/face_recognition/known_faces/Hameem/Hameem_2.jpg")
aaron_face_locations = face_recognition.face_locations(known_image_aaron)
aaron_face_landmarks_list = face_recognition.face_landmarks(known_image_aaron)
aaron_face_encoding = face_recognition.face_encodings(known_image_aaron)[0]
# print(aaron_face_locations)
# print(aaron_face_landmarks_list)



unknown_image_aaron = face_recognition.load_image_file("/home/knnan/Development/face_recognition/known_faces/Hameem/Hameem_4.jpg")
unown_aaron_face_locations = face_recognition.face_locations(unknown_image_aaron)
unown_aaron_face_landmarks_list = face_recognition.face_landmarks(unknown_image_aaron)
unown_aaron_face_encoding = face_recognition.face_encodings(unknown_image_aaron)[0]

results = face_recognition.compare_faces([aaron_face_encoding], unown_aaron_face_encoding)
print(results)
if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
