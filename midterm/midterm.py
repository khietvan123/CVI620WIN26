import cv2
import numpy as np
import os

# Load the Haar cascade classifiers for face and smile detection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_path  = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
smile_path = os.path.join(BASE_DIR, "haarcascade_smile.xml")

normal_face  = cv2.CascadeClassifier(face_path)
smiling_face = cv2.CascadeClassifier(smile_path)

if normal_face.empty() or smiling_face.empty():
    raise FileNotFoundError("Cascade XML not loaded. Check filenames + location.")
# Start video capture from the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened(): # Check if the video stream is opened successfully
    print("Error: Could not open video stream.")
    exit()

# Create a directory to save cropped images if it doesn't exist
output_dir = 'cropped_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize a counter for saving images
image_count = 0

# Capture video frames in a loop
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # OPTIONAL -> Flip the camera for mirror effect
    if not ret:
        print("Error: Could not read frame.")
        break

# ================= Face Detection ==================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = normal_face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w] # Define the ROI for smile detection
        roi_color = frame[y:y + h, x:x + w] # Define the color ROI for drawing rectangles

# ================= Smile Detection ==================
        smiling_faces = smiling_face.detectMultiScale(roi_gray, 1.8, 20) # Detect smiles within the face ROI
        if len(smiling_faces) > 0: # If smiles are detected
            label = "Smiling :)"
            for (sx, sy, sw, sh) in smiling_faces:
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

                # Save the cropped face with smile
                cropped_face = roi_color
                cv2.imwrite(os.path.join(output_dir, f'cropped_face_{image_count}.png'), cropped_face) # Save the image
                image_count += 1

        label = "Not Smiling :(" if len(smiling_faces) == 0 else "Smiling :)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Face and Smile capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
