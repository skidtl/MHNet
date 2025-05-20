from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./ultralytics/cfg/models/MHNet.yaml")
    result = model.train(data='ultralytics/cfg/datasets/VisDrone.yaml',
            imgsz=640,
            epochs=300,
            batch=4,
            project='runs/train',
            name='exp',
            resume = True
            )