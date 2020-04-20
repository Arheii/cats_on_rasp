import cv2
import numpy as np
import os
from datetime import datetime


class YoloNet(object):
    def __init__(self, vers='v3_medium', sens=0.5):
        if vers == 'v3_medium':
            net_weight = 'yolov3.weights'
            net_cfg = 'yolov3.cfg'
        elif vers == 'v3_tiny':
            net_weight = 'yolov3-tiny.weights'
            net_cfg = 'yolov3-tiny.cfg'
        self.vers = vers
        self.sens = sens
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.indexes = []
        self.cat = []   # coords and confidens for cats object
        self.person = []
        self.cat_raw = [] # need test few cats
        self.person_raw = [] # need test few men

        # Load Yolo,label for classes and output_layers
        models_dir = os.path.join(os.path.abspath(os.curdir), 'yolov3')
        net_weight = os.path.join(models_dir, net_weight)
        net_cfg = os.path.join(models_dir, net_cfg)
        coco_names = os.path.join(models_dir, 'coco.names')

        self.net = cv2.dnn.readNet(net_weight, net_cfg)
        with open(coco_names) as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))


    def recognition(self, file=False, frame=None):
        """recognition objects and their rectangle"""
        if file:
            frame = cv2.imread(file)
        self.frame = frame
        height, width, chanels = frame.shape

        # prepare frame and detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.indexes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.sens:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    self.boxes.append([x, y, w, h])
                    self.confidences.append(float(confidence))
                    self.class_ids.append(class_id)
 
        # some magic, delete doubles
        self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
        # print(self.indexes)
        id_cat = 15 #self.classes.index('cat')
        id_person = 0 #self.classes.index('person')
        n = len(self.boxes)
        self.cat_raw = [[self.boxes[i], self.confidences[i]] for i in range(n) if self.class_ids[i] == id_cat]
        self.person_raw = [[self.boxes[i], self.confidences[i]]  for i in range(n) if self.class_ids[i] == id_person]
        
        self.cat = [[self.boxes[i], self.confidences[i]]  for i in range(n) if self.class_ids[i] == id_cat and i in self.indexes]
        self.person = [[self.boxes[i], self.confidences[i]]  for i in range(n) if self.class_ids[i] == id_person and i in self.indexes]


    def frame_with_inf(self):
        """return frame with visual information"""
        font = cv2.FONT_ITALIC
        for i in range(len(self.boxes)):
            if i in self.indexes:
                x, y, w, h = self.boxes[i]
                label = str(self.classes[self.class_ids[i]])
                color = self.colors[i]
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.frame, label, (x, y + 30), font, 2, color, 3)
        return self.frame


if __name__ == '__main__':
    yn = YoloNet(vers='v3_tiny')
    f_name = '/home/pi/sv/datasets/img_from_pi/IMG_20200415_144949.jpg'
    start_time = datetime.now()
    yn.recognition(file=f_name)
    img = yn.frame_with_inf()
    f_name_inf = f_name[:-4] + '_inf.jpg'
    cv2.imwrite(f_name_inf, img)

    pred_time = datetime.now() - start_time
    cat = f'net_rec.cat={yn.cat}\n' if yn.cat else ''
    raw_cats = f'rawcat={yn.cat_raw}\n' if yn.cat_raw else ''
    person = f'net_rec.person={yn.person}\n' if yn.person else ''
    raw_person = f'rawperson={yn.person_raw}\n' if yn.person_raw else ''
    text_2 = f"time_spend: {pred_time}\n{cat}{raw_cats}{person}{raw_person}"
    print(text_2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()