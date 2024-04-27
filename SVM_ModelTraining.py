# Importing the required libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from DataCollection import DATA_PATH, actions, no_videos, frames_per_video
import pickle

# Main function
if __name__ == "__main__":
    # DATA LABELLING
    label_map = {label:num for num, label in enumerate(actions)}  # {'hello': 0, 'ok': 1, ..}

    videos = []    # To store the FEATURES
    labels = []    # To store the LABELS

    for action in actions:
        for video in range(no_videos):
            frames = []     # To store the all the 30 FRAMES of a single VIDEO
            for frame_num in range(frames_per_video):
                res = np.load(os.path.join(DATA_PATH, action, str(video), f"{frame_num}.npy"), encoding='ASCII')  # Loading the frame
                frames.append(res)

            videos.append(np.array(frames).flatten())               # Appending the features of a single video
            labels.append(label_map[action])    # Appending the label of the video
    

    # DATA PREPROCESSING
    X = np.array(videos)                    # Converting the FEATURES to numpy array
    Y = np.array(labels).reshape(-1,1)      # Converting the LABELS to numpy array

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=2, shuffle=True)  # Splitting the data into TRAIN and TEST sets


    # MODEL TRAINING
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)


    # EVALUATING THE MODEL
    Y_predicted = model.predict(X_test)  # Predicting the labels for the test data
    print(Y_test.flatten())
    print(Y_predicted)
    
    print("\n\n\nCLASSIFICATION REPORT\n", classification_report(Y_test, Y_predicted))  # Generate classification report

    print("\n\n\nACCURACY SCORE\n", accuracy_score(Y_test, Y_predicted))  # Accuracy Score


    # SAVING THE MODEL - TO USE IT LATER
    with open('SVM_Model.pkl', 'wb') as f:     # Saving the model in the current directory with the name 'SVM_Model.pkl'
        pickle.dump(model, f)