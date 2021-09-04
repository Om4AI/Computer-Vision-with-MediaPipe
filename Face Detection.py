import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture("Videos/4.mp4")
ptime = 0

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDet = mpFace.FaceDetection()

while True:
    ret, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceDet.process(imgrgb)

    # FPS
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,"FPS: "+str(int(fps)),
                (70,50),cv2.FONT_HERSHEY_PLAIN,
                3,(255,0,255), 2)

    if (results.detections):
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img,detection,
            #                       mpDraw.DrawingSpec(color=(0,0,255)),
            #                       mpDraw.DrawingSpec(color=(0,255,0)))
            bbox = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bboxC = int(bbox.xmin * w), int(bbox.ymin * h),\
                   int(bbox.width * w), int(bbox.height * h)
            cv2.rectangle(img, bboxC,(0,255,0),thickness=2)
            cv2.putText(img,str(int(detection.score[0]*100))+"%",
                        (bboxC[0], bboxC[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (0,255,0), 2)





    # Show results
    img = cv2.resize(img, (1480, 700))
    cv2.imshow("LiveStream", img)
    if (cv2.waitKey(1)==27):break