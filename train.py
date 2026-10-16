from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11/yolo11-obb-rapconv.yaml")
    train_results = model.train(
        data="ultralytics/cfg/datasets/coco_ence.yaml",
        epochs=200,
        imgsz=1024,
        batch=10,
        workers=8,
        device="0",
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        seed=0,
        deterministic=True,
        pretrained=False,
        amp=False,
        verbose=True,
        patience=200,
        iou=0.5,
        max_det=300,
        project="run/fair-csar",
        name="v11",
    )
