import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace

# Function to analyze facial attributes using DeepFace
def analyze_frame(frame):
    try:
        result = DeepFace.analyze(
            img_path=frame,
            actions=["age", "gender", "race", "emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            align=True,
            silent=True,
        )
        return result
    except Exception as e:
        return None

# Function to overlay text on the frame
def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.7  # Adjust transparency of overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0) # Add overlay on frame

    text_position = 30  # Initial text position
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        text_position += 25 # Increment text position for next text overlay

    return frame 

# Function to perform real-time face sentiment analysis
def facesentiment():
    cap = cv2.VideoCapture(0)  # Open webcam
    stframe = st.image([])  # Streamlit frame for display
    stop_button = st.button("Stop Webcam")  # Button to stop the webcam

    while not stop_button:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            st.error("Failed to access the webcam.")
            break

        # Convert frame to RGB (DeepFace requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze the frame using DeepFace
        result = analyze_frame(frame_rgb)

        print(result)

        if result:
            # Extract face bounding box
            face_coordinates = result[0]["region"]
            x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepare overlay texts
            texts = [
                f"Age: {result[0]['age']}",
                f"Gender: {result[0]['dominant_gender']} ({result[0]['gender'][result[0]['dominant_gender']]:.2f})",
                f"Race: {result[0]['dominant_race']} ({result[0]['race'][result[0]['dominant_race']]:.2f})",
                f"Emotion: {result[0]['dominant_emotion']} ({result[0]['emotion'][result[0]['dominant_emotion']]:.2f})",
            ]

            frame_rgb = overlay_text_on_frame(frame_rgb, texts)
        else:
            # Display a message when no face is detected
            texts = ["No face detected."]
            frame_rgb = overlay_text_on_frame(frame_rgb, texts)

        # Display the frame in Streamlit
        stframe.image(frame_rgb, channels="RGB")

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Main function for Streamlit app
def main():
    st.sidebar.title("Real-Time Face Analysis")
    activities = ["Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    st.sidebar.markdown(
        """Developed by Vinit keshri  
        [github.com/vinitkesh](https://github.com/vinitkesh)
        """
    )

    if choice == "Webcam Face Detection":
        st.markdown(
            """
            <div style="background-color:#6D7B8D;padding:10px">
            <h4 style="color:white;text-align:center;">
            Real-time face emotion recognition using OpenCV, DeepFace, and Streamlit.</h4>
            </div>
            <br>
            """,
            unsafe_allow_html=True,
        )
        facesentiment()

    elif choice == "About":
        st.subheader("About this App")
        st.markdown(
            """
            <div style="background-color:#98AFC7;padding:10px">
            <h4 style="color:white;text-align:center;">
            This Application is developed by Vinit Keshri.</h4>
            </div>
            <br>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
