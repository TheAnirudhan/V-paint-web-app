import cv2 as cv
import mediapipe as mp
import time


class handDetector:
    def __init__(self,mode=False,max_hands=2,dConfidence=0.5,tConfidence=0.5) -> None:
        self.mode = mode
        self.max_hands = max_hands
        self.dConfidence = dConfidence
        self.tConfidence = tConfidence
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.dConfidence, self.tConfidence)
    
    def findHands(self, img, draw=True):
        self.imgRGB  = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(self.imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img,handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def get_pos(self, img, handNo=0, draw=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            self.myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(self.myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img,(cx,cy),20,(200,0,255),-1)
        return lmList    


if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    Ptime = 0
    Ctime = 0
    detector = handDetector(False,2,0.7,0.3)
    while True:
        ret, img = cap.read()
        # FPS
        Ctime = time.time()
        fps = 1//(Ctime-Ptime)
        Ptime = Ctime 
        cv.putText(img,str(fps),(10,50), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        
        img = detector.findHands(img)
        lmHand = detector.get_pos(img,0,True)
        if len(lmHand)!=0:
            print(lmHand[0])

        # Display Image
        cv.imshow('Hand',img)

        if cv.waitKey(1) == 27:
            break