import cv2
import numpy as np
import tensorflow as tf

# Load the model with the custom layer
loaded_model = tf.keras.models.load_model("gender_classification.keras")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess frames
def preprocess_frame(frame, face_cascade):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # If no faces detected, return original frame
    if len(faces) == 0:
        return frame, None
    # Get coordinates of the first face
    (x, y, w, h) = faces[0]
    # Crop the frame to focus on the face
    face_frame = frame[y:y+h, x:x+w]
    # Resize the face to 64x64
    resized_face = cv2.resize(face_frame, (64, 64))
    # Return resized face frame and coordinates
    return resized_face, (x, y, w, h)

# Function to perform inference
def infer_gender(frame):
    # Normalize pixel values to range [0, 1]
    frame = frame / 255.0
    # Expand dimensions to match model input shape
    frame = np.expand_dims(frame, axis=0)
    # Perform inference
    result = loaded_model.predict(frame)
    return result[0][0]

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame to focus on the face and resize it to 64x64
    face_frame, face_coords = preprocess_frame(frame, face_cascade)

    # Perform inference on the face frame if it's detected
    if face_coords is not None:
        # Perform gender inference on the resized face frame
        prediction = infer_gender(face_frame)
        print(prediction)

        # Display the results on the frame
        
        if prediction <= 0.8:
            gender = 'female'
        else:
            gender ='male'
        (x, y, w, h) = face_coords
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    cv2.imshow('Gender Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
