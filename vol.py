
from ultralytics import YOLO


if __name__ == '__main__':                                                                                                                                                                  
    model = YOLO('best.pt')
    model.val(data='ultralytics/cfg/datasets/VisDrone.yaml',
              split='val',
              imgsz=640,
              batch=4,
              )
