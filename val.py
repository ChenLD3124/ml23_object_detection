from yolov5 import val
from yolov5.models.yolo import Model
import torch
val.run(data='yolo_data_6chan/yolo.yaml',batch_size=128,imgsz=400,task='test',weights='yolov5/runs/train/exp84/weights/best.pt')