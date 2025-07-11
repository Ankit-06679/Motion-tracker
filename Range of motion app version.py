import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # End point
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Application settings
st.title("Real-Time Joint Angle Measurement")
st.sidebar.title("Settings")
run_app = st.sidebar.checkbox("Run App")
st.sidebar.markdown("Use the checkbox to start/stop the app.")

# Live video feed
video_placeholder = st.empty()

# Joint angle tracking
joint_angles = {"Left Elbow": [], "Right Elbow": [], "Left Shoulder": [], "Right Shoulder": []}
timestamps = []

# Matplotlib figure for the graph
fig, ax = plt.subplots()
graph_placeholder = st.empty()

def plot_graph():
    ax.clear()
    for joint, angles in joint_angles.items():
        if len(angles) > 0:
            ax.plot(angles, label=joint)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Joint Angles Over Time")
    ax.legend(loc="upper right")
    ax.grid()

def main():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_counter = 0

        while run_app:
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor the frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Recolor back to BGR for OpenCV display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate angles
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Define joints to measure
                joints = {
                    "Left Elbow": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST],
                    "Right Elbow": [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST],
                    "Left Shoulder": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW],
                    "Right Shoulder": [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
                }

                for joint_name, landmark_ids in joints.items():
                    try:
                        p1 = [landmarks[landmark_ids[0].value].x, landmarks[landmark_ids[0].value].y]
                        p2 = [landmarks[landmark_ids[1].value].x, landmarks[landmark_ids[1].value].y]
                        p3 = [landmarks[landmark_ids[2].value].x, landmarks[landmark_ids[2].value].y]
                        angle = calculate_angle(p1, p2, p3)

                        # Append to joint_angles
                        joint_angles[joint_name].append(angle)

                        # Overlay the angle on the video
                        position = np.multiply([landmarks[landmark_ids[1].value].x, landmarks[landmark_ids[1].value].y], [640, 480]).astype(int)
                        cv2.putText(image, f"{joint_name}: {int(angle)}", tuple(position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except:
                        continue

                # Append timestamps
                timestamps.append(frame_counter)

            # Render pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the video in the app
            video_placeholder.image(image, channels="BGR", use_column_width=True)

            # Update frame counter
            frame_counter += 1

            # Update graph
            plot_graph()
            with graph_placeholder:
                st.pyplot(fig)

    cap.release()

if run_app:
    main()
