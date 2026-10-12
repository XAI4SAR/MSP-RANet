from ultralytics import YOLO
 
# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11-obb-rapconv.yaml")
# Train the model
train_results = model.train(
    data="ultralytics/cfg/datasets/coco_ence_csar.yaml",  
    epochs=200,  
    batch=12,
    amp=False,
    imgsz=1024,  
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    pretrained=False,
    verbose=True,
    optimizer="SGD",
    project='runs',##权重保存地址
    name='train',
    patience = 200,
    resume=True,
)
 

