import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)

# Hand Detection model
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

linedrawspecs = mpDraw.DrawingSpec(color=(0,255,0))
circledrawspecs = mpDraw.DrawingSpec(color=(0,0,255))

# FPS
ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the image in the hands function
    results = hands.process(imgRGB)

    # Extract to check if we have multiple hands
    if (results.multi_hand_landmarks):
        # Each hand
        for handlmks in results.multi_hand_landmarks:
            # Get Information
            for id, lm in enumerate(handlmks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
            
                if (id==8):
                    cv2.circle(img, (cx, cy), 17, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handlmks, mpHands.HAND_CONNECTIONS,circledrawspecs, linedrawspecs)
    
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, ("FPS: "+str(int(fps))), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)

    cv2.imshow("Livestream",img)
    if cv2.waitKey(1)==27:break
