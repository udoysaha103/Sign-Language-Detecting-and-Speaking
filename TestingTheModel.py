# Importing the required libraries
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import pyttsx3
from DataCollection import actions, mp_holistic, mp_detection, extract_keypoints


# Function to visualize the probabilities of each action
def probability_visualisation(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


if __name__ == "__main__":

    # MODEL INITIALIZATION

    # Using the 'Sequential' model and adding the LSTM and Dense layers
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(actions.shape[0], activation="softmax"))  # Output layer with 'softmax' activation function, makes sure the output is a probability distribution sums to 1


    # Compiling the model with 'Adam' optimizer
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])


    # LOADING THE PRE-TRAINED MODEL
    model.load_weights('mymodel.keras')


    # TESTING THE MODEL IN REAL-TIME
    colors = [(245,117,16), (117,245,16), (16,117,245), (196,117,245)]  # BGR colors for the rectangles of each action

    tts = pyttsx3.init()            # Text-to-Speech engine initialization
    tts.setProperty('rate', 125)    # Speed of the speech slowed down to normal

    frame_roll = []                 # To store the consecutive frames
    last_word = ""                  # To store the last predicted word
    threshold = 0.8                 # Threshold for the prediction
    res = []                        # To store the prediction probabilities

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():       # While the webcam is opened
            ret, frame = cap.read() # ret = True if the frame is read successfully, frame = the frame read from the webcam

            image, results = mp_detection(frame, holistic)  # Landmarks detection. image = Original Image, results = Landmarks

            # draw_landmarks(image, results)    # Drawing Landmarks on the image in-place

            keypoints = extract_keypoints(results)  # Extracting the keypoints from the landmarks
            frame_roll.append(keypoints)            # Appending the keypoints to the frame_roll
            
            if len(frame_roll) >= 30:   # If the frame_roll has 30 frames
                # Calculate the probabilities of each action
                res = model.predict(np.expand_dims(frame_roll[:30], axis=0))[0]

                if res[np.argmax(res)] > threshold:   # Action found
                    print(actions[np.argmax(res)])

                    if actions[np.argmax(res)] != last_word:    # If a new word is predicted
                        last_word = actions[np.argmax(res)]     # Update the last_word
                        tts.say(last_word)                      # Feed the word to the Text-to-Speech engine
                        tts.runAndWait()                        # Speak the word
                    
                    frame_roll = frame_roll[30:]      # Clear the first 30 frames, aka the Action
                else:
                    frame_roll = frame_roll[1:]       # Remove the first frame, as the action is not found

            image = probability_visualisation(res, actions, image, colors)      # Visualizing the probabilities of each action

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)          # Rectangle to display the last predicted word
            cv2.putText(image, last_word+" "+str(len(frame_roll)), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)    # Displaying the last predicted word

            # Display the frame
            cv2.imshow("Sign language detection", image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        
        # Release the webcam and destroy the windows
        cap.release()
        cv2.destroyAllWindows()