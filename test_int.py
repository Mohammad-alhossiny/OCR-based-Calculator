from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer
import torch
import json
import cv2
from detc import JersyDetector


conf = open('dect.json')
data = json.load(conf)
conf.close()
jersey_detector = JersyDetector(data)
update = data["update"]
weights = data["weights"]

def tensors(d_num):
    img = cv2.imread(f'drawings/drawn{d_num}.png')
    img = cv2.resize(img, (1216, 416))

    with torch.no_grad():
      if update:  # update all models (to fix SourceChangeWarning)
          for weights in ['yolov7.pt']:
              return(jersey_detector.detect(img))
              strip_optimizer(weights)
      else:
          return(jersey_detector.detect(img))

