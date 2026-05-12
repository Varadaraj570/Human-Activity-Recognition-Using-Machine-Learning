import cv2
import mediapipe as mp
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

label = input("Enter activity (standing/sitting/walking): ")

file = open('../dataset/har_data.csv', 'a', newline='')
writer = csv.writer(file)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        row = []
        for lm in landmarks:
            row.append(lm.x)
            row.append(lm.y)

        row.append(label)
        writer.writerow(row)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()