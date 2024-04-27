# Importing the required libraries
import numpy as np
import cv2
import pickle
import pyttsx3
from DataCollection import actions, mp_holistic, mp_detection, extract_keypoints, draw_landmarks


if __name__ == "__main__":

    # LOADING THE PRE-TRAINED MODEL
    with open('RandomForest_model.pkl', 'rb') as f:
        model = pickle.load(f)


    # TESTING THE MODEL IN REAL-TIME

    tts = pyttsx3.init()            # Text-to-Speech engine initialization
    tts.setProperty('rate', 125)    # Speed of the speech slowed down to normal

    frame_roll = []                 # To store the consecutive frames
    last_word = ""                  # To store the last predicted word

    # add a stabilizer if possible

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():       # While the webcam is opened
            ret, frame = cap.read() # ret = True if the frame is read successfully, frame = the frame read from the webcam

            image, results = mp_detection(frame, holistic)  # Landmarks detection. image = Original Image, results = Landmarks

            draw_landmarks(image, results)    # Drawing Landmarks on the image in-place

            keypoints = extract_keypoints(results)  # Extracting the keypoints from the landmarks
            frame_roll.append(keypoints)            # Appending the keypoints to the frame_roll
            
            if len(frame_roll) >= 30:   # If the frame_roll has 30 frames
                # Calculate the probabilities of each action
                res = model.predict(np.expand_dims(np.array(frame_roll[:30]).flatten(), axis=0))[0]

                print(actions[res])

                if actions[res] != last_word:    # If a new word is predicted
                    last_word = actions[res]        # Update the last_word
                    tts.say(last_word)              # Feed the word to the Text-to-Speech engine
                    tts.runAndWait()                # Speak the word
                
                frame_roll = frame_roll[1:]       # Remove the first frame

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)          # Rectangle to display the last predicted word
            cv2.putText(image, last_word, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)    # Displaying the last predicted word

            # Display the frame
            cv2.imshow("Sign language detection", image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        
        # Release the webcam and destroy the windows
        cap.release()
        cv2.destroyAllWindows()