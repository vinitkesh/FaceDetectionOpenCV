# OpenCV Web based Emotion Detector
- This project implements real-time facial emotion detection using the `deepface` library and OpenCV.
- It captures video from the webcam, detects faces, and predicts the emotions associated with each face. The emotion labels are displayed on the frames in real-time.
- to implement realtime emotion monitoring.
- Created a streamLit application for the facial emotion recognition of human faces.

## Dependencies
- [deepface](https://github.com/serengil/deepface): A deep learning facial analysis library that provides pre-trained models for facial emotion detection. It relies on TensorFlow for the underlying deep learning operations.
- [OpenCV](https://opencv.org/): An open-source computer vision library used for image and video processing.

## Usage
- Clone the repository and navigate to the project directory.
- Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

- Run the `app.py` script to start the real-time emotion detection application:
```bash
streamlit app.py
```

- The application will open in a new browser window, displaying the webcam feed with emotion labels overlaid on the detected faces.

## Implementation

### Emotion Detection

- The `deepface` library provides a pre-trained model for facial emotion detection. The model is based on a Convolutional Neural Network (CNN) architecture and is trained on the FER2013 dataset, which contains images of faces labeled with seven different emotions: angry, disgust, fear, happy, sad, surprise, and neutral.

- The `DeepFace.analyze()` function is used to detect the emotions of faces in an image. It takes an image as input and returns a dictionary containing the emotion predictions for each face detected in the image.

- The `detect_emotions()` function processes the webcam feed frame by frame, detects faces using the Haar cascade classifier, and predicts the emotions of each face using the `deepface` model. The emotion labels are then overlaid on the frame using OpenCV drawing functions.

### Real-time Video Capture

- The `cv2.VideoCapture()` function is used to capture video from the webcam. The video feed is read frame by frame, and the `detect_emotions()` function is called on each frame to detect and display the emotions of faces in real-time.

- The `cv2.imshow()` function is used to display the processed frames with emotion labels overlaid. The `cv2.waitKey()` function is used to wait for a key press to exit the application.

- The `cv2.destroyAllWindows()` function is called to close all OpenCV windows and release the webcam resources when the application is exited.

  ## StreamLit
- Deployment on StreamLit framework.
- ![teat](ryan.png)
- 
- 

