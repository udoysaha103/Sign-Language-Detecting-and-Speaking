# Importing Libraries
import numpy as np
import cv2
import os
import mediapipe as mp


# GLOBAL VARIABLES -----------------------------------------------------------------------------------------------------------------------------------
DATA_PATH = os.path.join("DATASET")         # Path to the DATASET

# Actions/Classes in the dataset (Sign Language)
actions = np.array(["hello", "OK"])                  # <-- Add the desired actions for Sign Detection

no_videos = 100                                      # 100 sequences/videos per action
frames_per_video = 30                                # 30 frames per sequence/video

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
# -----------------------------------------------------------------------------------------------------------------------------------------------------


# Detecting Face, Pose, Left & Right Hand LANDMARKS using MediaPipe Holistic
def mp_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # By default, OpenCV reads images in BGR format. So we need to convert it to RGB format.
    image.flags.writeable = False
    results = model.process(image)                   # Detecting Landmarks in the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # Convert the image back to BGR format
    return image, results                            # Image = Original Image, Results = Landmarks


# Drawing Landmarks on the Image in-place
def draw_landmarks(image, results):
    # Face Landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    
    # Pose Landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    
    # Left & Right Hand Landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    

# Extracting Keypoints from the Landmarks
def extract_keypoints(results):
    # Pose has 33 Landmarks, Face has 468 Landmarks, Left & Right Hand has 21 Landmarks
    # (33*4)+(468*3)+(21*3)+(21*3) = 1662 keypoints in total
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# Main Function
if __name__ == "__main__":

    # CREATE THE DATASET FOLDER STRUCTURE
    for action in actions:
        for video in range(no_videos):
            try:
                # Under the DATA_PATH, there will be a folder for each action. Under each action folder, there will be a folder for each video.
                # Ex: DATA_PATH/hello/1, DATA_PATH/hello/2, DATA_PATH/hello/3, ... , DATA_PATH/OK/1, DATA_PATH/OK/2, DATA_PATH/OK/3, ...
                os.makedirs(os.path.join(DATA_PATH, action, str(video)))
            except:
                pass
    

    # COLLECTING THE DATA
    cap = cv2.VideoCapture(0)

    # For each action, 'no_videos' number of videos will be collected. Each video will have 'frames_per_video' number of frames.
    # Ex: For 'hello' action, 100 videos will be collected. Each video will have 30 frames.
    #     Each frame will have 1662 keypoints. So, the shape of the dataset will be (100, 30, 1662) = 4986000 keypoints for a single action.

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for video in range(no_videos):
                for frame_num in range(frames_per_video):
                    # Read the frame from the webcam
                    ret, frame = cap.read()                          # ret = True if the frame is read successfully, frame = the frame read from the webcam

                    # Detecting Landmarks in the frame using 'holistic' model
                    image, results = mp_detection(frame, holistic)   # image = Original Image, results = Landmarks

                    # Drawing Landmarks on the image in-place
                    draw_landmarks(image, results)


                    # Displaying data collection message on screen
                    if video == 0:      # Show 'Starting collection for ACTION' for a new Action
                        cv2.putText(image, f"Starting collection for {action.upper()}", (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'For: {action}, Video no: {video}, Frame no: {frame_num}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        cv2.imshow("Collecting frames", image)
                        cv2.waitKey(10000)  # Wait for 10 seconds before starting the collection
                    elif video == 30 or video == 70:      # Show 'Rest for ACTION' for a new Action
                        cv2.putText(image, f"Rest for {action.upper()}", (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'For: {action}, Video no: {video}, Frame no: {frame_num}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        cv2.imshow("Collecting frames", image)
                        cv2.waitKey(10000)  # Wait for 10 seconds before starting the collection
                        
                    if frame_num == 0:      # Show 'STARTING COLLECTION' for a new Video
                        cv2.putText(image, "STARTING COLLECTION", (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'For: {action}, Video no: {video}, Frame no: {frame_num}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        cv2.imshow("Collecting frames", image)
                        cv2.waitKey(2000)   # Wait for 2 seconds before starting the collection
                    else:
                        cv2.putText(image, f'For: {action}, Video no: {video}, Frame no: {frame_num}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        cv2.imshow("Collecting frames", image)


                    # Extracting Keypoints from the Landmarks
                    keypoints = extract_keypoints(results)

                    # Saving the keypoints in a .npy file
                    npy_path = os.path.join(DATA_PATH, action, str(video), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Beaking gracefully
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break
        
        # Releasing the webcam and closing the windows after the data collection is done
        cap.release()
        cv2.destroyAllWindows()