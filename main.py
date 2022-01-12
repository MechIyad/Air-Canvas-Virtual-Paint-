import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,460)

pts = []
prevs = [[0,0]]
blank = np.zeros((480,640,3), np.uint8)
xp, yp = 0, 0
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

def a(a):
    pass

cv2.namedWindow("TRACK BAR")
cv2.resizeWindow("TRACK BAR",640,240)
cv2.createTrackbar("RED","TRACK BAR",0,255, a)
cv2.createTrackbar("GREEN","TRACK BAR",0,255, a)
cv2.createTrackbar("BLUE","TRACK BAR",255,255,a)
cv2.createTrackbar("LINE THIKNESS","TRACK BAR",2,40, a)




def detect_hand(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    return results

def return_keypoint_location_from_id(img, results, idx):
    for lms in results.multi_hand_landmarks:
        for id, lm in enumerate(lms.landmark):
            if id==idx:
                return (int(lm.x*img.shape[1]), int(lm.y*img.shape[0]))

while cap.isOpened():
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    r = cv2.getTrackbarPos("RED","TRACK BAR")
    b = cv2.getTrackbarPos("BLUE","TRACK BAR")
    g = cv2.getTrackbarPos("GREEN","TRACK BAR")
    t = cv2.getTrackbarPos("LINE THIKNESS","TRACK BAR")
    if t==0:
        t=1
    dis = img.copy()
    results = detect_hand(img)
    if not suc:
        print("unabel to get camera feed")
        break
    key = cv2.waitKey(1)

    if key == ord("c"):
        pts = []
        xp, yp = 0, 0
        blank[:]=(0, 0, 0)
        print("clear")


    elif key==ord("e"):
        if results.multi_hand_landmarks:
            mpDraw.draw_landmarks(dis,results.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)
            d_finger = return_keypoint_location_from_id(img, results, 8)
            m_finger = return_keypoint_location_from_id(img, results, 12)

            if d_finger[1]<m_finger[1]:
                cv2.circle(dis,d_finger, t, (110,22,245), -1)
                if xp==0 and yp==0:
                    xp, yp = d_finger
                cv2.line(blank, (xp, yp), d_finger, (0, 0, 0), t)
                xp, yp = d_finger
            else:
                xp, yp = 0, 0

    else:
        
        if results.multi_hand_landmarks:
            mpDraw.draw_landmarks(dis,results.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)
            d_finger = return_keypoint_location_from_id(img, results, 8)
            m_finger = return_keypoint_location_from_id(img, results, 12)

            if d_finger[1]<m_finger[1]:
                cv2.circle(dis,d_finger, t, (110,22,245), -1)
                if xp==0 and yp==0:
                    xp, yp = d_finger
                cv2.line(dis, (xp, yp), d_finger, (b, g, r), t)
                cv2.line(blank, (xp, yp), d_finger, (b, g, r), t)
                xp, yp = d_finger
            else:
                xp, yp = 0, 0
    dis = cv2.addWeighted(dis, 0.7, blank, .5, 0)
    cv2.imshow("IMAGE", dis)
    cv2.imshow("PAPER", blank)
    cv2.waitKey(1)
    if key==27:
        break
    elif key==ord("s"):
        print("saving")
        cv2.imwrite('results/from camera.png', dis)
        cv2.imwrite('results/black canvas.png', blank)
