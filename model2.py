import cv2
import dlib

# Load the face and eye detectors
face_detector = dlib.get_frontal_face_detector()
eye_tracker = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Capture video from the webcam
cap = cv2.VideoCapture(0)

def calculate_ear(eye_landmarks):
    # Calculate the Euclidean distances between the vertical eye landmarks
    vertical_dist1 = cv2.norm(eye_landmarks[1], eye_landmarks[5])
    vertical_dist2 = cv2.norm(eye_landmarks[2], eye_landmarks[4])

    # Calculate the Euclidean distance between the horizontal eye landmarks
    horizontal_dist = cv2.norm(eye_landmarks[0], eye_landmarks[3])

    # Calculate the eye aspect ratio (EAR)
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)

    return ear

# Define the threshold for eye closure detection
EAR_THRESHOLD = 0.2

# Counter for closed eye detections
closed_eye_counter = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        # Detect eye landmarks in the face
        landmarks = eye_tracker(gray, face)

        # Extract the coordinates of the left and right eye landmarks
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate the eye aspect ratio (EAR) for each eye
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Draw rectangles around the eyes
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.rectangle(frame, (min(left_eye)[0], min(left_eye, key=lambda x: x[1])[1]),
                      (max(left_eye)[0], max(left_eye, key=lambda x: x[1])[1]), (0, 0, 255), 2)
        cv2.rectangle(frame, (min(right_eye)[0], min(right_eye, key=lambda x: x[1])[1]),
                      (max(right_eye)[0], max(right_eye, key=lambda x: x[1])[1]), (0, 0, 255), 2)

        # Check if the eyes are closed
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            #cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            closed_eye_counter += 1
            if closed_eye_counter >= 3:
                cv2.putText(frame, "Sleep Danger!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            closed_eye_counter = 0

    # Display the frame
    cv2.imshow('Eye Tracking', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

