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

# Initialize graph data
joint_angles = {}
timestamps = []
joints = {
    "Neck": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.NOSE.value],
    "Left Shoulder": [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value],
    "Right Shoulder": [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value],
    "Left Elbow": [mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.LEFT_INDEX.value],
    "Right Elbow": [mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value, mp_pose.PoseLandmark.RIGHT_INDEX.value],
    "Left Hip": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
    "Right Hip": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    "Left Knee": [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value],
    "Right Knee": [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value],
    "Left Ankle": [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_HEEL.value],
    "Right Ankle": [mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_HEEL.value],
}
for joint in joints.keys():
    joint_angles[joint] = []

# Define colors for each joint
joint_colors = {
    "Neck": (255, 0, 0),
    "Left Shoulder": (0, 255, 0),
    "Right Shoulder": (0, 0, 255),
    "Left Elbow": (255, 255, 0),
    "Right Elbow": (255, 0, 255),
    "Left Hip": (0, 255, 255),
    "Right Hip": (128, 128, 0),
    "Left Knee": (128, 0, 128),
    "Right Knee": (0, 128, 128),
    "Left Ankle": (128, 128, 255),
    "Right Ankle": (255, 128, 128),
}

# Capture video
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detect landmarks
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate angles for each joint
            for joint_name, landmarks_ids in joints.items():
                try:
                    p1 = [landmarks[landmarks_ids[0]].x, landmarks[landmarks_ids[0]].y]
                    p2 = [landmarks[landmarks_ids[1]].x, landmarks[landmarks_ids[1]].y]
                    p3 = [landmarks[landmarks_ids[2]].x, landmarks[landmarks_ids[2]].y]

                    angle = calculate_angle(p1, p2, p3)
                    joint_angles[joint_name].append(angle)

                    # Display angle on the frame with color coding
                    cv2.putText(image, f'{joint_name}: {int(angle)}', 
                                tuple(np.multiply(p2, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, joint_colors[joint_name], 2)
                except:
                    continue

            # Add timestamp for plotting
            timestamps.append(len(timestamps))

        # Render landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display frame
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Plot graph
def update_graph(i):
    plt.clf()
    for joint, angles in joint_angles.items():
        plt.plot(timestamps[:len(angles)], angles, label=joint)

    plt.xlabel('Time (frames)')
    plt.ylabel('Angle (degrees)')
    plt.title('Real-Time Joint Angles')
    plt.legend(loc='upper right')
    plt.grid()

ani = FuncAnimation(plt.gcf(), update_graph, interval=100)
plt.show()