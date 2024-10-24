import cv2
import mediapipe as mp
from math import sqrt, acos, degrees
import time
import math
 
# Initialize Mediapipe Hand model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define connections between landmarks
connections = [[0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
               [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
               [0, 9], [9, 10], [10, 11], [11, 12],  # Middle finger
               [0, 13], [13, 14], [14, 15], [15, 16],  # Ring finger
               [0, 17], [17, 18], [18, 19], [19, 20],  # Little finger
               [17, 4], [20, 8]  # connections between mk and lasts
               ]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize variables for threat detection
threat_open_hand = False
threat_close_hand = False
start_time = None

# Initialize stack for saving threat messages
threat_stack = []

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        # Read frame from video capture
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe Hand model
        results = hands.process(image)

        # Convert the RGB image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        Length_26 = 0.0
        Length_20 = 0.0
        Length_3 = 0.0
        Length_7 = 0.0
        Length_11 = 0.0
        Length_15 = 0.0
        Length_19 = 0.0
        ecart = 0.0

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Get the bounding box coordinates of the hand region
            x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
            for landmark in results.multi_hand_landmarks[0].landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Check if hand is within the fixed box
            if x_min > 100 and y_min > 100 and x_max < image.shape[1] and y_max < image.shape[0]:
                # Draw the green box around the hand region
                cv2.rectangle(image, (350, 350), (100, 100), (0, 255, 0), 2)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                # Draw lines connecting consecutive landmarks and display length
                for i, connection in enumerate(connections):
                    x0, y0 = results.multi_hand_landmarks[0].landmark[connection[0]].x, results.multi_hand_landmarks[0].landmark[connection[0]].y
                    x1, y1 = results.multi_hand_landmarks[0].landmark[connection[1]].x, results.multi_hand_landmarks[0].landmark[connection[1]].y

                    # Calculate the length between landmarks
                    length =  abs(y1 - y0) * bbox_height
                    if i == 20 :
                        length = (math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / bbox_width) * 1000
                        Length_20 = length
                    if i == 25 :
                        length = (math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / bbox_width) * 1000
                        Length_26 = length                    
                    # Draw line on the image
                    cv2.line(image, (int(x0 * image.shape[1]), int(y0 * image.shape[0])),
                             (int(x1 * image.shape[1]), int(y1 * image.shape[0])), (0, 0, 255), 2)

                    if i in [7, 11, 15, 19]:
                    # Display the length text
                        length_text = f"L{i}: {length:.2f}"
                        cv2.putText(image, length_text, (int(x0 * image.shape[1]), int(y0 * image.shape[0]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        if i == 3:
                            Length_3 =  length 
                        elif i == 7:
                            Length_7 =  length 
                        elif i == 11:
                            Length_11 = length 
                        elif i == 15:
                            Length_15 = length 
                        elif i == 19:
                            Length_19 = length 

                # Calculate angles and segments
                angles = []
                segments = []

                for i in range(1, len(connections)):
                    x0, y0 = results.multi_hand_landmarks[0].landmark[connections[i][0]].x, results.multi_hand_landmarks[0].landmark[connections[i][0]].y
                    x1, y1 = results.multi_hand_landmarks[0].landmark[connections[i][1]].x, results.multi_hand_landmarks[0].landmark[connections[i][1]].y

                    # Calculate the vectors AB and BC
                    vector_AB = (x0 - x1, y0 - y1)
                    magnitude_AB = sqrt(vector_AB[0] ** 2 + vector_AB[1] ** 2)

                    # Calculate the angle between vectors AB and BC
                    if i < len(connections) - 1:
                        x2, y2 = results.multi_hand_landmarks[0].landmark[connections[i + 1][1]].x, \
                                 results.multi_hand_landmarks[0].landmark[connections[i + 1][1]].y
                        vector_BC = (x2 - x1, y2 - y1)
                        magnitude_BC = sqrt(vector_BC[0] ** 2 + vector_BC[1] ** 2)
                        dot_product = vector_AB[0] * vector_BC[0] + vector_AB[1] * vector_BC[1]
                        angle = acos(dot_product / (magnitude_AB * magnitude_BC))
                        angles.append(angle)

                        # Threat detection logic
                        if threat_open_hand and threat_close_hand:
                            cv2.putText(image, "Danger !!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            if start_time is None:
                                start_time = time.time()
                            elif time.time() - start_time > 5:
                                start_time = None
                                threat_open_hand = False
                                threat_close_hand = False
                        if (Length_20 < 0.62):
                            if (( Length_19 < 7) and (Length_15< 7)  and (Length_11 < 7)  and (Length_7 < 7) ):
                                threat_close_hand = True
                                cv2.putText(image, "Threat Sign close hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 255, 0), 2)
                            else :
                                threat_open_hand = True
                                cv2.putText(image, "Threat Sign open hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 255, 0), 2)
                                

        # Display the image
        cv2.imshow('Hand Tracking', image)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
