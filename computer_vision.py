import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

ptime = 0
ctime = 0
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils



while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpdraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img, str(int(fps)), (10, 70),
    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


