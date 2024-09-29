import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-8e96f-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-8e96f.appspot.com"  # No gs:// prefix
})

# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []

# Load images and upload to Firebase Storage
for path in pathList:
    img = cv2.imread(os.path.join(folderPath, path))
    if img is None:
        print(f"Warning: Image {path} could not be loaded.")
        continue  # Skip this image if it cannot be loaded
    imgList.append(img)
    studentIds.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()  # Use the initialized bucket
    blob = bucket.blob(fileName)

    try:
        blob.upload_from_filename(fileName)
        print(f"Uploaded {fileName} to Firebase Storage.")
    except Exception as e:
        print(f"Error uploading {fileName}: {e}")

print(studentIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if any encodings were found
            encodeList.append(encodings[0])
        else:
            print("Warning: No face found in an image.")
    return encodeList


print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
if not encodeListKnown:
    print("Error: No encodings found. Exiting.")
    exit(1)  # Exit if no encodings are found

encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

# Save the encoding to a file
try:
    with open("EncodeFile.p", 'wb') as file:
        pickle.dump(encodeListKnownWithIds, file)
    print("File Saved")
except Exception as e:
    print(f"Error saving file: {e}")