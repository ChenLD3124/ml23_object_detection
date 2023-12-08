from yolov5 import train
import time
train.run(data='yolo_data_6chan/yolo.yaml', imgsz=400, cfg='yolov5/models/myyolov5n.yaml',weights='myyolov5n2.ptt',hyp='yolov5/data/hyps/hyp.scratch-low.yaml',batch_size=128,epochs=50)