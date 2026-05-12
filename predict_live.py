import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("../models/har_model.pkl")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('../results/output_demo.avi',
                      fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640,480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    activity = "Detecting..."

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        row = []
        for lm in landmarks:
            row.append(lm.x)
            row.append(lm.y)

        X = np.array(row).reshape(1, -1)

        prediction = model.predict(X)[0]
        activity = prediction

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.rectangle(frame, (10,10), (450,70), (0,0,0), -1)

    cv2.putText(frame, f'Activity: {activity}', (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    cv2.imshow("ML Human Activity Recognition", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()