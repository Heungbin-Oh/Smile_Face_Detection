import cv2
import os
from datetime import datetime

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

# Create directory to save smile images in real time
output_dir = os.path.join(os.path.dirname(__file__), "smile_faces")
os.makedirs(output_dir, exist_ok=True)

# 1. Capture video frames from a webcam.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame
    if not ret:
        print("Failed to grab frame.")
        break

# 2. Convert frames to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 3. Detect faces using haarcascade_frontalface_default.xml.
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 4. Crop each detected face region
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]  # colored one for saving

        # 5. Detect smiles within the cropped face region using haarcascade_smile.xml
        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=30,
            minSize=(25, 25)
        )

        # 6. Draw rectangles (Blue) around detected faces and smiles
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # draw rectangles (Green) for smiles
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_color, (sx, sy),
                          (sx + sw, sy + sh), (0, 255, 0), 2)

        # 7. Display results in real time and save cropped faces with smiles to disk
        if len(smiles) > 0:
            label = "Smiling :)"
            color = (0, 255, 0)
            # Save the smiling face
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = os.path.join(output_dir, f"smile_{timestamp}.jpg")
            cv2.imwrite(filename, face_color)
        else:
            label = "No smile"
            color = (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the resulting frame
    cv2.imshow('Real Time Face and Smile Detector', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
