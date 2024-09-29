import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://faceattendancerealtime-8e96f-default-rtdb.firebaseio.com/",
        "storageBucket": "faceattendancerealtime-8e96f.appspot.com",
    },
)

bucket = storage.bucket()

# Load the encoding file
print("Loading Encode File ...")
file = open("EncodeFile.p", "rb")
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")
print(f"Student IDs in encode file: {studentIds}")  # Debug print

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the background image
imgBackground = cv2.imread("Resources/background.png")

# Load mode images
folderModePath = "Resources/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))


def process_detected_face(id):
    # Get the Data
    studentInfo = db.reference(f"Students/{id}").get()

    if studentInfo is None:
        print(f"Student with ID {id} not found in the database")
        return "Unknown", 0

    try:
        # Update data of attendance
        datetimeObject = datetime.strptime(
            studentInfo["last_attendance_time"], "%Y-%m-%d %H:%M:%S"
        )
        secondsElapsed = (datetime.now() - datetimeObject).total_seconds()

        if secondsElapsed > 30:
            ref = db.reference(f"Students/{id}")
            studentInfo["total_attendance"] += 1
            ref.child("total_attendance").set(studentInfo["total_attendance"])
            ref.child("last_attendance_time").set(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        return studentInfo["name"], studentInfo["total_attendance"]
    except KeyError as e:
        print(f"Error processing student {id}: Missing key {e}")
        return "Error", 0
    except Exception as e:
        print(f"Unexpected error processing student {id}: {e}")
        return "Error", 0


def display_attendance_list(imgBackground, attendance_list):
    # Clear the previous list
    cv2.rectangle(imgBackground, (1000, 44), (1280, 677), (255, 255, 255), -1)

    # Display header
    cv2.putText(
        imgBackground,
        "Attendance List",
        (1020, 80),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (0, 0, 0),
        1,
    )
    cv2.line(imgBackground, (1000, 90), (1280, 90), (0, 0, 0), 1)

    # Display attendance list
    start_y = 120
    for i, (name, attendance) in enumerate(attendance_list):
        y = start_y + i * 30
        cv2.putText(
            imgBackground,
            f"{name}: {attendance}",
            (1020, y),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    return imgBackground


attendance_list = []

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162 : 162 + 480, 55 : 55 + 640] = img
    imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[0]

    attendance_list.clear()  # Clear the list for each frame

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]
                print(f"Detected face with ID: {id}")  # Debug print

                # Process each detected face
                name, attendance = process_detected_face(id)
                attendance_list.append((name, attendance))

    # Display attendance list
    imgBackground = display_attendance_list(imgBackground, attendance_list)

    cv2.imshow("Face Attendance", imgBackground)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
