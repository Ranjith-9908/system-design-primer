import cv2
import dlib
from scipy.spatial import distance

def calculate_eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Define the thresholds for eye aspect ratio and consecutive frames
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

# Initialize the frame counters and drowsiness flag
COUNTER = 0
ALARM_ON = False

# Load the facial landmark predictor from dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Path to the dlib shape predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Start the video stream
video_stream = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the path to a video file

while True:
    # Read a frame from the video stream
    ret, frame = video_stream.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = detector(gray, 0)
    
    for face in faces:
        # Predict the facial landmarks for the face
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract the left and right eye coordinates
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        
        # Calculate the eye aspect ratios
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        
        # Average the eye aspect ratios
        ear = (left_ear + right_ear) / 2.0
        
        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            
            # If the eyes have been closed for a sufficient number of frames, set the alarm flag
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    # Perform any desired actions (e.g., sound an alarm)
        
        else:
            COUNTER = 0
            ALARM_ON = False
        
        # Draw the computed eye aspect ratio on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # Display the resulting frame
    cv2.imshow("Drowsiness Detection", frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
video_stream.release()
cv2.destroyAllWindows()
