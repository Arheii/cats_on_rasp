# script based on https://proglib.io/p/real-time-object-detection/
# my model notebook https://colab.research.google.com/drive/1xyp2CZ_5-cpIkFAqcYY9NRcx1x61JYy9

import numpy as np
import time
import cv2
import os


MODEL = 'resnet50_cats.onnx'
LABELS = ['cat', 'human', 'nothing']


class ClassificatorNet(object):
    def __init__(self):
        self.percents = False
        self.mark =  False
        #load net
        path=os.path.join(os.path.abspath(os.curdir) , MODEL)
        self.net = cv2.dnn.readNetFromONNX(path) 
    
    def recognition(self, file=False, frame=None):
        # pass the blob through the network and obtain the detections and
        # predictions
        if file:
            frame = cv2.imread(file)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)),scalefactor=1.0/224,
                                     size=(224, 224), mean= (104, 117, 123), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()
        #maybe it's softmax, mb no (: 
        sm = cv2.exp(out[0]) * 100 / sum(cv2.exp(out[0]))[0]
        sm = [x[0] for x in sm]
        self.percents = list(zip(LABELS, sm))
        self.mark = np.argmax(out)
        return self.percents[self.mark]


if __name__ == '__main__':
    net_cl = ClassificatorNet()
    img_file = '/home/pi/sv/datasets/img_from_pi/0102.jpg'
    frame = cv2.imread(img_file)
    categ, perc = net_cl.recognition(frame=frame)
    print(net_cl.percents)
    cv2.putText(frame, f'it is {categ} with {perc:.2f}%', (30,30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 230, 220), 2)
    cv2.imshow("Web camera view", frame)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()


