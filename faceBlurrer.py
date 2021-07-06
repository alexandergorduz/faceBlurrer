import cv2
import mediapipe
import numpy as np

cap = cv2.VideoCapture(0)

mpFacesMeshes = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=5)

while True:
    _, frame = cap.read()
    frameCopy = frame.copy()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mpFacesMeshes.process(image)

    if result.multi_face_landmarks:
        for faceLandmarks in result.multi_face_landmarks:
            landmarksCoords = [[[int(faceLandmark.x * frame.shape[1]), int(faceLandmark.y * frame.shape[0])]] for faceLandmark in faceLandmarks.landmark]
            convexhull = cv2.convexHull(np.array(landmarksCoords))
            mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
            cv2.fillConvexPoly(mask, convexhull, 255)
            frameCopy = cv2.blur(frameCopy, (27, 27))
            faceExtracted = cv2.bitwise_and(frameCopy, frameCopy, mask=mask)
            backgroundMask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=backgroundMask)
            frame = cv2.add(background, faceExtracted)
    
    cv2.imshow('output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()