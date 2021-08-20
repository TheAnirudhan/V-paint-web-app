import cv2 as cv
import mediapipe as mp
import time
import handlm
import numpy as np


class vPaint():
    def __init__(self) -> None:
        self.cap = cv.VideoCapture(0)
        self.ret, self.img = self.cap.read()
        self.screen = cv.cvtColor(np.zeros(self.img.shape, dtype = np.uint8), cv.COLOR_RGB2BGR)
        self.pointer = cv.cvtColor(np.zeros(self.img.shape, dtype = np.uint8), cv.COLOR_RGB2BGR)
        self.template = cv.cvtColor(np.zeros(self.img.shape, dtype = np.uint8), cv.COLOR_RGB2BGR)

        self.prevpt = None
        self.opentime = 0
        self.closetime = 0
        self.Ptime = 0
        self.Ctime = 0
        self.detector = handlm.handDetector(False,1,0.7,0.5)
    def __del__(self):
        self.cap.release()

    def cal_length(self,p1,p2):
        return(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

    def handsClosed(self, lmHand):
        l1 = self.cal_length(lmHand[8][1:], lmHand[5][1:])
        l2 = self.cal_length(lmHand[12][1:], lmHand[9][1:])
        l3 = self.cal_length(lmHand[16][1:], lmHand[13][1:])
        l4 = self.cal_length(lmHand[20][1:], lmHand[17][1:])
        
        if(  np.mean([l1,l2,l3,l4]) <25): return 1
        else: return 0

    def get_frame(self):
        ret, frame = self.cap.read()
        ret, jpeg = cv.imencode('.jpg',frame)
        return jpeg.tobytes()

    def get_board(self):
        ret, self.img = self.cap.read()
        self.img = cv.flip(self.img,1)
        self.pointer = cv.cvtColor(np.zeros(self.img.shape, dtype = np.uint8), cv.COLOR_RGB2BGR)
        self.img = self.detector.findHands(self.img)
        lmHand = self.detector.get_pos(self.img)
        if len(lmHand)!=0:
            self.thumbtip = (lmHand[4][1:])
            self.indextip = (lmHand[8][1:])
            self.middletip = (lmHand[12][1:])

            self.length = self.cal_length(self.indextip,self.middletip)
            # print(length)

            if(self.length>45):
                if self.prevpt == None:
                    cv.circle(self.screen,self.indextip,5,(255,255,0),-1)
                else:
                    cv.line(self.screen,self.prevpt,self.indextip,(255,255,0),5)
                self.template = self.screen
            else:
                cv.circle(self.pointer,self.indextip,7,(1000,100,100),-1)
                self.template = self.screen + self.pointer
            self.prevpt = self.indextip

            
            if not self.handsClosed(lmHand):
                self.opentime = time.time()
            if(self.handsClosed(lmHand)):
                self.closetime = time.time()
                
                if((self.closetime-self.opentime) > 1.5):
                    self.screen = cv.cvtColor(np.zeros(self.img.shape, dtype = np.uint8), cv.COLOR_RGB2BGR)

        img = np.hstack([self.template, self.img])
        ret, self.jpeg1 = cv.imencode('.jpg',img)
        return self.jpeg1.tobytes()
