# cats_on_rasp
Telegram bot and neutral networks for recognition photos in raspberry pi 

Telegram bot allows you to remotely start and stop recording on the raspberry, as well as immediately get the predictions of one of the three neural networks: based in ResNet5, Yolo (406dpi) and Yolo Tiny.  
 
Used cv2, telegram bot and PiCamera frameworks.  
If you want to use yolo(406dpi) net,  you need download https://pjreddie.com/media/files/yolov3.weights (250mb) and put it to yolov3 folder.

This is my part of a joint project (using Bitbucket), which allows a raspberry to record any movement, as well as track and predict the position of cats to point a water cannon at them.
