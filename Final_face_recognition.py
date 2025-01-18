import face_recognition
import cv2

known_face_encodings = []
known_face_names = []

known_p1_img = face_recognition.load_image_file("C:/Users/hs089/Downloads/elon.jpeg")
known_p2_img = face_recognition.load_image_file("c:/Users/hs089/OneDrive/Desktop/srk.jpg")

known_p1_encoding = face_recognition.face_encodings(known_p1_img)[0]
known_p2_encoding = face_recognition.face_encodings(known_p2_img)[0]

known_face_encodings.append(known_p1_encoding)
known_face_encodings.append(known_p2_encoding)

known_face_names.append("Elon Musk")
known_face_names.append("Shahrukh Khan")

face_capture = cv2.CascadeClassifier("C:/Users/hs089/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# To capture video from webcam.
video_capture = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, video = video_capture.read()
    video = cv2.flip(video,1)

    col = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

    face = face_capture.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_locations =  face_recognition.face_locations(video)
    face_encodings = face_recognition.face_encodings(video, face_locations)

    for (top,right,bottom,left), face_encodings in zip(face_locations,face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle around the faces
        for (x, y, w, h) in face:
            cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw label with name below the face
        cv2.rectangle(video, (left, bottom - 35), (right, bottom), (0, 0, 255))
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(video, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow("Face Detection and Recognition", video)

    # Break the loop
    if cv2.waitKey(10) == 27:  # Press 'Esc' key to exit
        break


# Release the VideoCapture object
video_capture.release()
cv2.destroyAllWindows()
