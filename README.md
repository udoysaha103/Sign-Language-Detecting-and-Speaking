
# Sign Language Detecting and Speaking

This project aims to recognize sign language gestures using webcam input and speak out the detected word in real-time. 


## Demo

.


## Installing dependencies

To run the program, make sure to install the packages using the following line from the terminal:

```bash
  pip install numpy opencv-python mediapipe scikit-learn tensorflow pyttsx3
```

If you have a GPU, run this command also. This step is optional.

```bash
  pip install tensorflow-gpu
```
## User guide

1. Run `DataCollection.py` to collect the dataset for training.

2. Run `ModelTraining.py` to train the AI model.

3. Finally, run `TestingTheModel.py` to test the trained model in real-time.




## Project description

It consists of three main programs:

1. **DataCollection.py**: This program collects the data required for training the sign language recognition model. It accesses the camera device to capture frames, extracts facial and hand landmarks using MediaPipe Holistic, and stores the keypoints in a numpy array. The dataset creation process is lightweight and efficient, making the execution faster.

- **NB:** *New actions can be added by modifying `line 12` of this file.*

The dataset is saved in the `DATASET` directory.

2. **ModelTraining.py**: This program trains the AI model for sign language recognition and saves the trained model in the `mymodel.keras` file. It utilizes a Sequential model with LSTM and Dense layers to produce a probability distribution for each action. The action with the highest probability is considered the output action.
- **NB:** The number of epochs for training can be adjusted from `line 57` of this file based on the model's performance. 
The training log, including error and accuracy information, is stored in the `LOGS` folder. The model is also evaluated within this program.

3. **TestingTheModel.py**: This program loads the trained model and performs real-time sign language recognition using webcam input. It predicts the sign language gestures and utilizes the `pyttsx3` module to speak out the predicted word in real-time.


## Badges

Open access to anyone!

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
## Contributing

Contributions are always welcome!



## Special attribution

Took inspiration from https://github.com/nicknochnack/ActionDetectionforSignLanguage.git.
## 🚀 About Me

This is Udoy Saha. I am tech enthusiast, problem solver.

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://udoysaha.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/udoysaha103/)


## Feedback

If you have any feedback, please reach out to us at udoysaha103@gmail.com.

