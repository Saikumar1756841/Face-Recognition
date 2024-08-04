# Import required modules
import cv2
import os
import numpy as np

# Define the subjects (persons) for face recognition
subjects = ["", "sai kumar", "harsha", "vamshi"]

# Function to detect face using OpenCV
# Function to detect face using OpenCV
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    if len(faces) == 0:
        return None, None
    
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


# Function to prepare training data
def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(data_folder_path, dir_name)  # Use os.path.join to create directory path
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = os.path.join(subject_dir_path, image_name)  # Use os.path.join to create image path
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    
    return faces, labels

# Prepare training data
data_folder_path = r"C:\Users\annad\Downloads\opencv-face-recognition-python-master\opencv-face-recognition-python-master\training-data"
print("Preparing data...")
faces, labels = prepare_training_data(data_folder_path)
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# Train face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
print("Training completed")

# Function to predict the face using the trained model
def predict(img):
    img_copy = img.copy()
    face, rect = detect_face(img_copy)
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img_copy, rect)
    draw_text(img_copy, label_text, rect[0], rect[1]-5)
    return img_copy

# Function to draw rectangle on image
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Function to draw text on image
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Start capturing video from camera
print("Starting camera...")
video_capture = cv2.VideoCapture(0)

# Main loop to capture live images and perform prediction
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture image from camera")
        break
    
    predicted_frame = predict(frame)
    cv2.imshow("Face Recognition", predicted_frame)
    
    # Press 'q' to exit the live face recognition loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
