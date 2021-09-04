import time
import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


ptime = 0
cap = cv2.VideoCapture('Pose Estimation Videos\production ID_5192157.mp4')
linedrawspecs = mpDraw.DrawingSpec(color=(0,255,0))
circledrawspecs = mpDraw.DrawingSpec(color=(0,0,255))

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if (results.pose_landmarks):
        mpDraw.draw_landmarks(img, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS,
                              circledrawspecs, linedrawspecs)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            # Get the pixel values
            cx, cy = int(lm.x*w), int(lm.y*h)
            # Check the pixels
            if (id==30): cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)


    # FPS
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    # cv2.putText(img, "FPS: "+str(fps), (70,50), cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)

    img = cv2.resize(img, (1280, 700))  # Resize image
    cv2.imshow("Video", img)
    if cv2.waitKey(1)==27: break