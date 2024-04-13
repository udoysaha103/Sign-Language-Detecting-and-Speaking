# Importing the required libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from DataCollection import DATA_PATH, actions, no_videos, frames_per_video


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
                res = np.load(os.path.join(DATA_PATH, action, str(video), f"{frame_num}.npy"))  # Loading the frame
                frames.append(res)

            videos.append(frames)               # Appending the features of a single video
            labels.append(label_map[action])    # Appending the label of the video
    

    # DATA PREPROCESSING
    X = np.array(videos)                    # Converting the FEATURES to numpy array
    Y = to_categorical(labels).astype(int)  # Converting the LABELS to one-hot encoded numpy array

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)  # Splitting the data into TRAIN and TEST sets


    # MODEL TRAINING
    LOG_DIR = os.path.join("LOGS")                  # Path to store the model training logs - including the loss and accuracy
    tb_callback = TensorBoard(log_dir=LOG_DIR)      # TensorBoard callback to store the logs

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

    # Training the model with the training data
    model.fit(X_train, Y_train, epochs=1000, callbacks=[tb_callback])   # <-- Training the model for 500 epochs, changing the number of epochs can be done based on the model performance

    # Summary of the model - Model Architecture
    print(model.summary())

    # MODEL TRAINING COMPLETE


    # EVALUATING THE MODEL
    Y_predicted = model.predict(X_test)  # Predicting the labels for the test data

    actual = np.argmax(Y_test, axis=1).tolist()
    results = np.argmax(Y_predicted, axis=1).tolist()
    print("\n\n\nCONFUSION MATRIX\n", multilabel_confusion_matrix(actual, results))  # Confusion Matrix

    print("\n\n\nACCURACY SCORE\n", accuracy_score(actual, results))  # Accuracy Score


    # SAVING THE MODEL - TO USE IT LATER
    model.save('mymodel.keras')     # Saving the model in the current directory with the name 'mymodel.keras'